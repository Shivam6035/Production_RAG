import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# 1. Load Environment Variables safely
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Grab the Direct Connection String instead of the REST URL
DB_CONNECTION_STRING = os.getenv("SUPABASE_DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def connect_to_supabase() -> PGVector: 
    """Connect to Supabase pgvector using the direct database connection."""
    
    if not DB_CONNECTION_STRING:
        raise ValueError("SUPABASE_DATABASE_URL is missing. Please add it to your .env file.")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please add it to your .env file.")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=GEMINI_API_KEY,
    )

    # Initialize the PGVector store
    vectorstore = PGVector( 
        embeddings=embeddings, 
        collection_name="production_docs",
        connection=DB_CONNECTION_STRING, 
        use_jsonb=True,
    )
    
    return vectorstore

def verify_connection(vectorstore: PGVector) -> bool: 
    """Verify the connection works by writing and retrieving a test document.""" 
    
    test_doc = Document( 
        page_content="This is a test document to verify Supabase pgvector.",
        metadata={"test": True} 
    )
    
    try:
        # 1. Test Writing
        ids = vectorstore.add_documents([test_doc])
        print(f"✅ Verified connection and added test document. ID: {ids[0]}")

        # 2. Test Reading/Searching
        results = vectorstore.similarity_search("test document")
        if results:
            print(f"🔍 Search works! Found: '{results[0].page_content}'")

        # 3. Cleanup (Optional, but recommended for tests)
        # vectorstore.delete(ids) 
        # print("🧹 Cleanup complete.")    

        return True

    except Exception as e: 
        print(f"❌ Database Interaction Error: {e}") 
        return False
    
def main():
    print("=" * 60)
    print("Supabase pgvector Connection Test (LangChain)")
    print("=" * 60)

    if not DB_CONNECTION_STRING: 
        print("❌ No Supabase Database URL found.")
        print("Please ensure SUPABASE_DB_URL is defined in your .env file.")
        return

    try:
        # Safely parse the host for the print statement 
        # (Handles format: postgresql+psycopg://user:pass@host:port/db)
        host_part = DB_CONNECTION_STRING.split('@')[-1].split(':')[0]
        
        print(f"\n🔗 Connecting to Supabase Database...") 
        print(f"📍 Host: {host_part}") 
        
        # Run the connection and verification
        vector_store = connect_to_supabase()
        verify_connection(vector_store)
        
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")

if __name__ == "__main__":
    main()