
import langchain
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
# from ollama import embeddings
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


recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, 
    chunk_overlap=50, 
    separators=['\n\n', '\n', '. ', ' '] 
)

recursive_chunks = recursive_splitter.split_text(document)


print(f"Recursive chunks: {len(recursive_chunks)}") 
for i, chunk in enumerate(recursive_chunks) : 
    print(f"\\n --- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:100]+"..." if len(chunk) > 100 else f"\\n --- Chunk {i+1} ---")


# semantic splitting using the SemanticChunker class, 
# which uses the embedding model to determine semantic breakpoints in the text.

semantic_chunker = SemanticChunker(
    embeddings_model,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90  # Split at 90th percentile
)



semantic_chunks = semantic_chunker.split_text(document) # passing document

print(f"Semantic chunks: {len(semantic_chunks)}") 
for i, chunk in enumerate(semantic_chunks) : 
    print(f"\\n --- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:100]+"..." if len(chunk) > 100 else f"\\n --- Chunk {i+1} ---")


# Create two vector stores - one for each chunking method

recursive_vectorstore = Chroma.from_texts( 
    recursive_chunks, 
    embeddings_model, 
    collection_name='recursive_chunks')

semantic_vectorstore = Chroma. from_texts( 
    semantic_chunks, 
    embeddings_model, 
    collection_name='semantic_chunks')

# Test queries 
test_queries = [ 
    'How do I authenticate with OAuth2?', 
    'What happens when I hit the rate limit?', 
    'How are webhooks secured?', 
    'What format are errors returned in?'
]

def test_retrieval(query, vectorstore, name) : 
    results = vectorstore.similarity_search(query, k=1) 
    print(f'\\n{name} - Query: \"{query}\"') 
    print(f'Retrieved: {results[0].page_content [ : 150] }... ' ) 
    return results [0].page_content

print(f"\n{'='*60}") 
print(" RETRIEVAL TESTS") 
print(f"{'='*60}")


for query in test_queries : 
    print('=' * 60) 
    recursive_result = test_retrieval(query, recursive_vectorstore, 'RECURSIVE') 
    semantic_result = test_retrieval(query, semantic_vectorstore, 'SEMANTIC' )

