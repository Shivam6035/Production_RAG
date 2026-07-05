# production ready

import langchain
from langchain import embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
# from ollama import embeddings
from ollama import embeddings
from langchain_experimental.text_splitter import SemanticChunker
load_dotenv()
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
# embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
import os
from dotenv import load_dotenv
# 1. Use the correct class name in the import
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# 2. Use the correct class name to initialize the model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2", 
    google_api_key=os.getenv("GEMINI_API_KEY")
)

document = '''
    # Authentication Guide 

    ## OAuth2 Authentication 

    # To authenticate with our API, you need OAuth2 credentials. 
    # First, obtain a client id and client_secret from the developer portal. 
    # Make a POST request to /oauth/token with grant type=client credentials.
    #  The response contains an access_token valid for 3600 seconds.
    # Include this token in the Authorization header as 'Bearer <token> !.

    ## Rate Limiting 
    # Our API implements rate limiting using a token bucket algorithm. 
    # Free tier: 100 requests per minute. Pro tier: 1000 requests per minute. 
    # Enterprise tier: Custom limits. When rate limited, you receive a 429 status code. 
    # The Retry-After header indicates when to retry.

    ## Error Handling 

    # All errors return a standard JSON format. 
    # The 'code' field contains a machine-readable error code. 
    # The 'message' field contains a human-readable description. 
    # Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST. 
    # Always check the HTTP status code first, then parse the error body.

    ## Webhooks Configure 
    # webhooks in your dashboard settings. 
    # We support-HTTP and HTTPS endpoints. 
    # Webhook payloads are signed with HMAC-SHA256. 
    # Verify signatures using your webhook secret. 
    # Failed deliveries are retried with exponential backoff.

    '''



def smart_chunker(
    text: str,
    use_semantic: bool = True,
    fallback_chunk_size: int = 500,
) -> list[str]:
    """
    Production-ready chunking.

    Uses SemanticChunker first and falls back to RecursiveCharacterTextSplitter
    if semantic chunking fails or produces oversized chunks.
    """

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    if use_semantic:
        try:
            chunker = SemanticChunker(
                embeddings=embeddings_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,
            )

            chunks = chunker.split_text(text)

            max_chunk_size = 2000

            if any(len(chunk) > max_chunk_size for chunk in chunks):
                print("Oversized semantic chunks detected. Falling back to recursive chunking.")
                return _recursive_fallback(text, fallback_chunk_size)

            return chunks

        except Exception as e:
            print(f"Semantic chunking failed: {e}")
            print("Using recursive fallback...")
            return _recursive_fallback(text, fallback_chunk_size)

    return _recursive_fallback(text, fallback_chunk_size)


def _recursive_fallback(text: str, chunk_size: int) -> list[str]:
    """
    Recursive fallback chunking.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
    )

    return splitter.split_text(text)


# -------------------------
# Example Usage
# -------------------------

chunks = smart_chunker(document, use_semantic=True)

print(f"Created {len(chunks)} chunks\n")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}")
    print("-" * 50)
    print(chunk)
    print()




