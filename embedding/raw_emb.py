import os
from google import genai
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()

# Retrieve your API key
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the official Google GenAI client
client = genai.Client(api_key=api_key)

# # Generate content using the new SDK syntax
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What is the capital of Italy?'
)

# Print the text response directly
print(response.text)