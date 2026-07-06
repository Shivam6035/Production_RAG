# import os
# from realtime import Optional, dataclass
# # import from realtimedataclass
# from pathlib import Path
# from dotenv import load_dotenv
# from langchain_postgres import PGVector 
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
# from langchain_core.prompts import ChatPromptTemplate 
# from langchain_core. runnables import RunnablePassthrough 
# from langchain_core.output_parsers import StrOutputParser 
# from langchain_core.documents import Document

# load_dotenv()  # Load environment variables from .env file

# @dataclass
# class Config:

# # Database - use pooler URL in production 
# # 
#     database_url: str = os.getenv(
#         "SUPABASE_DATABASE_URL",
#         os. getenv( "DATABASE_URL", 
#         "DATABASE_URL","postgresql://postgres: postgres@localhost :!5432/postgres"),


#     )
#     collection_name: str = "production_documents"
#     embedding_model: str = "gemini-embedding-2" 
#     chat_model: str = "gemini-2.5-flash"
#     temperature: float = 0.0


#     default_k: int = 5 
#     min_similarity: float = 0.5

# class RAGService: 
#     """Production-ready RAG service with pgvector"""

#     def __init__(self, config: Optional[Config] = None):
#         self.config = config or Config()
#         self._vectorstore = None
#         self._chain = None

#     @property
#     def vectorstore(self) -> PGVector:
#         """Lazy initialization of vectorstore"""
#         if self._vectorstore is None:
#             self._vectorstore = PGVector(
#                 connection_string=self.config.database_url,
#                 collection_name=self.config.collection_name,
#                 embedding_function=GoogleGenerativeAIEmbeddings(model=self.config.embedding_model),
              

#                 connection=self.config.database_url,
#                 use_jsonb=True,
#             )
#         return self._vectorstore
    

#     @property
#     def chain(self):
#         """Lazy initialization of RAG chain"""
#         if self._chain is None:
#             self._chain = self._create_chain()
#         return self._chain

#     def create_chain(self):
#         """Create the RAG chain"""
#         retriever = self.vectorstore.as_retriever(
#             search_kwargs={"k": self.config.default_k})
        

        

#         llm = ChatGoogleGenerativeAI( model=self.config.chat_model, 
#                                      temperature=self.config.temperature, 
#                                      api_key=os.getenv("GEMINI_API_KEY"))

#         prompt = ChatPromptTemplate.from_template(
#             """  
# Answer the question based ONLY on the following context.
# If the answer is not in the context, respond with: "I don't have information about that in my knowledge base."

# Context:
# {context}

# Question: {question}

# Answer:"""
#     )

#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)

#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     print("RAG with Fallback:\n")

#     questions = [
#         "What is the pricing for LangSmith?",  # In knowledge base
#         "What is the stock price of OpenAI?",  # Not in knowledge base
#         "How do I deploy LangChain to AWS?",  # Not in knowledge base
#     ]

#     for q in questions:
#         answer = rag_chain.invoke(q)
#         print(f"Q: {q}")
#         print(f"A: {answer}\n")


# def demo_structured_rag():
#     """RAG with structured output."""

#     vectorstore = create_kb()
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     class RAGResponse(BaseModel):
#         """Structured RAG response."""

#         answer: str = Field(description="The answer to the question")
#         confidence: str = Field(description="high, medium, or low")
#         sources_used: List[str] = Field(description="List of sources referenced")
#         follow_up: str = Field(description="Suggested follow-up question")

#     structured_llm = llm.with_structured_output(RAGResponse)

#     prompt = ChatPromptTemplate.from_template(
#         """
# Based on the context below, answer the question.

# Context:
# {context}

# Question: {question}

# Provide a structured response."""
#     )

#     def format_docs(docs):
#         return "\n\n".join(
#             f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
#             for doc in docs
#         )

#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | structured_llm
#     )
#     print("Structured RAG Demo:\n")
#     result = rag_chain.invoke("What is LangGraph?")

#     print(f"Answer: {result.answer}")
#     print(f"Confidence: {result.confidence}")
#     print(f"Sources: {result.sources_used}")
#     print(f"Follow-up: {result.follow_up}")


# # Exercise: Build a document Q&A system
# def exercise_document_qa():
#     """
#     EXERCISE: Build a complete document Q&A system that:
#     1. Takes a text document as input
#     2. Splits and embeds it
#     3. Allows multiple questions
#     4. Returns answers with confidence scores
#     """

#     class DocumentQA:
#         def __init__(self, document: str, source_name: str = "document"):
#             # Split document
#             splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#             doc = Document(page_content=document, metadata={"source": source_name})
#             chunks = splitter.split_documents([doc])

#             # Create vector store
#             self.vectorstore = Chroma.from_documents(
#                 documents=chunks,
#                 embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2"),
#             )
#             self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

#             # Create chain
#             self.llm = init_chat_model(model="gpt-4o-mini", temperature=0.2)

#             self.prompt = ChatPromptTemplate.from_template(
#                 """
# Answer based on the context. Rate your confidence (high/medium/low).

# Context: {context}
# Question: {question}

# Format: [Confidence: X] Answer"""
#             )

#             def format_docs(docs):
#                 return "\n".join(d.page_content for d in docs)

#             self.chain = (
#                 {
#                     "context": self.retriever | format_docs,
#                     "question": RunnablePassthrough(),
#                 }
#                 | self.prompt
#                 | self.llm
#                 | StrOutputParser()
#             )

#         def ask(self, question: str) -> str:
#             return self.chain.invoke(question)

#     # Test
#     test_doc = """
#     The Python programming language was created by Guido van Rossum.
#     First released in 1991, Python emphasizes code readability.
#     Python 3.12 was released in October 2023 with improved error messages.
#     The language is named after Monty Python, not the snake.
#     """

#     qa = DocumentQA(test_doc, "python_facts")

#     print("Document Q&A System:\n")
#     questions = [
#         "Who created Python?",
#         "When was Python 3.12 released?",
#         "Why is Python named Python?",
#     ]

#     for q in questions:
#         answer = qa.ask(q)
#         print(f"Q: {q}")
#         print(f"A: {answer}\n")


# if __name__ == "__main__":
#     # demo_basic_rag()
#     # demo_rag_with_sources()
#     # demo_rag_with_fallback()
#     # demo_structured_rag()
#     exercise_document_qa()