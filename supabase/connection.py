import os
from dotenv import load_dotenv
from supabase import create_client, Client

def main():
    # 1. Load environment variables from the .env file
    load_dotenv()
    
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    
    # Ensure the variables were actually found
    if not url or not key:
        print("❌ Error: Missing SUPABASE_URL or SUPABASE_KEY in .env file.")
        return

    try:
        # 2. Initialize the Supabase client
        supabase: Client = create_client(url, key)
        print("⏳ Attempting to connect to Supabase...")
        
        # 3. Check if the API key is live
        # We test this by making a lightweight request to list storage buckets.
        # If the key is dead or invalid, this will throw an authentication exception.
        supabase.storage.list_buckets()
        
        print("✅ Connection successful! Your Supabase API key is live and valid.")
        
        # --- You can now proceed with your database operations below ---
        # Example: data = supabase.table("your_table_name").select("*").execute()
        
    except Exception as e:
        # Catching the exception if the key is invalid or the URL is wrong
        print(f"❌ Connection failed! The API key might be invalid or expired.")
        print(f"Detailed Error: {e}")

if __name__ == "__main__":
    main()