# testing production

import os
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_postgres import PGVector 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from the correct directory layout
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Config:
    """Production Configuration Environment."""
    # Prioritizes direct connection pooled URL string
    DATABASE_URL: str = os.getenv(
        "SUPABASE_DB_URL", 
        os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/postgres")
    )
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    COLLECTION_NAME: str = "production_documents"
    
    # Using verified production-grade Google models
    EMBEDDING_MODEL: str = "gemini-embedding-2" 
    CHAT_MODEL: str = "gemini-2.5-flash"
    TEMPERATURE: float = 0.0
    DEFAULT_K: int = 3 

# Define Pydantic Structure for Structured Output
class RAGResponse(BaseModel):
    """Structured response definition for the LLM output layer."""
    answer: str = Field(description="The answer to the question based strictly on the context provided.")
    confidence: str = Field(description="Confidence grading score: high, medium, or low.")
    sources_used: List[str] = Field(description="List of sources or source metadata keys referenced.")
    follow_up: str = Field(description="Suggested relevant follow-up question based on the output payload.")


class RAGService: 
    """Production-ready RAG service infrastructure using Supabase pgvector."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        if not self.config.GEMINI_API_KEY:
            raise ValueError("❌ Missing GEMINI_API_KEY environment variable.")
        if "postgresql" not in self.config.DATABASE_URL:
            raise ValueError("❌ Invalid connection string format. Ensure it is a valid postgresql+psycopg URL.")

        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.EMBEDDING_MODEL,
            google_api_key=self.config.GEMINI_API_KEY
        )
        self._vectorstore = None
        self._llm = None

    @property
    def vectorstore(self) -> PGVector:
        """Lazy initialization of PGVector instance."""
        if self._vectorstore is None:
            self._vectorstore = PGVector(
                collection_name=self.config.COLLECTION_NAME,
                connection=self.config.DATABASE_URL, 
                embeddings=self._embeddings,
                use_jsonb=True,
            )
        return self._vectorstore

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Lazy initialization of generative LLM engine."""
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.config.CHAT_MODEL, 
                temperature=self.config.TEMPERATURE, 
                google_api_key=self.config.GEMINI_API_KEY
            )
        return self._llm

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Utility processing method to join raw documents together into a text block."""
        return "\n\n".join(f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}" for doc in docs)

    def run_fallback_rag(self, question: str) -> str:
        """Executes a standard string-in, string-out RAG pipeline with strict guardrails."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.DEFAULT_K})
        
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based ONLY on the following context.
If the answer is not contained in the context, respond explicitly with: "I don't have information about that in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""
        )

        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question)

    def run_structured_rag(self, question: str) -> RAGResponse:
        """Executes a RAG lookup returning structured corporate data payloads."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.DEFAULT_K})
        structured_llm = self.llm.with_structured_output(RAGResponse)

        prompt = ChatPromptTemplate.from_template(
            """Based strictly on the provided context context below, answer the question.

Context:
{context}

Question: {question}

Provide a structured response object matching the expected schema schema parsing layers."""
        )

        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | structured_llm
        )
        return rag_chain.invoke(question)

    def seed_documents(self, raw_text: str, source_name: str = "document_source"):
        """Utility method to safely split, process, and write new elements to pgvector."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        doc = Document(page_content=raw_text, metadata={"source": source_name})
        chunks = splitter.split_documents([doc])
        
        ids = self.vectorstore.add_documents(chunks)
        print(f"📦 Successfully seeded database with {len(ids)} document chunks.")
        return ids


# =====================================================================
# Production Testing Suite Executions
# =====================================================================

def main():
    print("=" * 60)
    print("🚀 Initializing Production RAG Engine Pipeline Architecture")
    print("=" * 60)

    try:
        # Initialize Service
        rag_service = RAGService()
        
        # 1. Seed sample pipeline test facts into Supabase Vectorstore
        test_knowledge_base = """
        LangSmith is an enterprise software optimization tool. Pricing starts at a Free Tier for developers, followed by a Developer Plan at $39/month, and custom Enterprise agreements.
        LangGraph is a sophisticated framework designed to build stateful, multi-agent applications using graph networks.
        The Python programming language was created by Guido van Rossum and first released natively in 1991.
        """
        print("\n[Step 1] Seeding production facts to Supabase Vector store...")
        seeded_ids = rag_service.seed_documents(test_knowledge_base, source_name="system_manifest_v1")

        # 2. Test Fallback Pipeline Architecture
        print("\n[Step 2] Executing Fallback RAG Chain Executions:")
        questions = [
            "What is the pricing for LangSmith?",  # Expected to find valid data
            "What is the current stock price of OpenAI?",  # Expected to hit prompt boundary limits
        ]

        for q in questions:
            ans = rag_service.run_fallback_rag(q)
            print(f"❓ Q: {q}")
            print(f"📥 A: {ans}\n")

        # 3. Test Structured Output Pipeline Architecture
        print("[Step 3] Executing Pydantic-Validated Structured RAG Architecture:")
        structured_query = "What is LangGraph?"
        
        result: RAGResponse = rag_service.run_structured_rag(structured_query)
        print(f"❓ Q: {structured_query}")
        print(f"✨ Parsed Object Output Properties:")
        print(f"   ↳ Answer: {result.answer}")
        print(f"   ↳ Confidence Metrics: {result.confidence}")
        print(f"   ↳ Traced Sources: {result.sources_used}")
        print(f"   ↳ Smart Follow-up: {result.follow_up}\n")

        # 4. Optional Storage Space Cleanup
        # print("[Step 4] Cleaning up generated test documents from table configurations...")
        # rag_service.vectorstore.delete(seeded_ids)
        # print("🧹 Table space cleared.")

    except Exception as e:
        print(f"\n❌ Production Runtime Core Interruption: {e}")

if __name__ == "__main__":
    main()