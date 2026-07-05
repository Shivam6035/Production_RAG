import os
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="langchain_community"
)

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever

# -----------------------------------------------------------------------------
# Load Environment Variables
# -----------------------------------------------------------------------------

load_dotenv()

# -----------------------------------------------------------------------------
# Initialize Embedding Model
# -----------------------------------------------------------------------------

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------------------------------------------------------
# Sample Documents
# -----------------------------------------------------------------------------

documents = [
    Document(
        page_content=(
            "Product SKU-7742X is our flagship router. "
            "It supports gigabit speeds and advanced QoS features."
        ),
        metadata={"type": "product"},
    ),
    Document(
        page_content=(
            "For network connectivity issues, first check the "
            "ethernet cable and router status lights."
        ),
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content=(
            "Error code E_CONN_REFUSED indicates the server "
            "rejected the connection. Check firewall settings."
        ),
        metadata={"type": "error"},
    ),
    Document(
        page_content=(
            "The authentication process requires valid credentials. "
            "Use authentication tokens or API keys as needed."
        ),
        metadata={"type": "authentication"},
    ),
    Document(
        page_content=(
            "Router configuration guide: Access the admin panel "
            "at 192.168.1.1 to modify settings."
        ),
        metadata={"type": "config"},
    ),
    Document(
        page_content=(
            "WCAG 2.1 compliance requires all images to have "
            "alt text and sufficient color contrast."
        ),
        metadata={"type": "compliance"},
    ),
]

print(f"\nLoaded {len(documents)} documents.")

# -----------------------------------------------------------------------------
# Create Chroma Vector Store
# -----------------------------------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    collection_name="hybrid_test",
)

vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Vector retriever created.")

# -----------------------------------------------------------------------------
# Create BM25 Retriever
# -----------------------------------------------------------------------------

bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3,
)

print("BM25 retriever created.")

# -----------------------------------------------------------------------------
# Custom Hybrid Retrieval using Weighted Reciprocal Rank Fusion (RRF)
# -----------------------------------------------------------------------------

def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    """
    Combine multiple retrievers using Weighted Reciprocal Rank Fusion.
    """

    doc_scores = {}

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)

        for rank, doc in enumerate(results):
            key = doc.page_content
            score = weight * (1.0 / (rank + rrf_k))

            if key in doc_scores:
                previous_score, _ = doc_scores[key]
                doc_scores[key] = (previous_score + score, doc)
            else:
                doc_scores[key] = (score, doc)

    ranked_docs = sorted(
        doc_scores.values(),
        key=lambda x: x[0],
        reverse=True,
    )

    return [doc for _, doc in ranked_docs[:k]]

# -----------------------------------------------------------------------------
# Display Results
# -----------------------------------------------------------------------------

def display_results(query, name, results):
    print(f"\n{name} Results")
    print("-" * 40)
    print(f"Query: {query}\n")

    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}\n")

# -----------------------------------------------------------------------------
# Test Queries
# -----------------------------------------------------------------------------

test_queries = [
    "SKU-7742X specifications",
    "E_CONN_REFUSED error",
    "How do I authenticate?",
    "WCAG compliance",
    "router configuration",
]

# -----------------------------------------------------------------------------
# Run Tests
# -----------------------------------------------------------------------------

for query in test_queries:

    print("\n" + "=" * 80)

    # -------------------------
    # Vector Search
    # -------------------------

    vector_results = vector_retriever.invoke(query)

    display_results(
        query=query,
        name="VECTOR SEARCH",
        results=vector_results,
    )

    # -------------------------
    # BM25 Search
    # -------------------------

    bm25_results = bm25_retriever.invoke(query)

    display_results(
        query=query,
        name="BM25 SEARCH",
        results=bm25_results,
    )

    # -------------------------
    # Hybrid Search (Custom RRF)
    # -------------------------

    hybrid_results = hybrid_retrieve(
        query=query,
        retrievers=[
            bm25_retriever,
            vector_retriever,
        ],
        weights=[
            0.5,
            0.5,
        ],
        k=3,
    )

    display_results(
        query=query,
        name="HYBRID SEARCH (Custom RRF)",
        results=hybrid_results,
    )