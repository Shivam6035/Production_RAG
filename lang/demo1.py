# from google import genai

# this is demo scrip to see tracking in langsmith

from google import genai
from langsmith import wrappers
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from google import genai

# 2. Load the environment variables from your .env file
load_dotenv()

def main():
    genai.Client() #reads GOOGLE_API_KEY / GEMINI_API_KEY from the environment
    gemini_client = genai.Client()

    # Wrap the Gemini client to enable LangSmith tracing
    client = wrappers.wrap_gemini(
        gemini_client,
        tracing_extra={
            "tags": ["gemini", "python"],
            "metadata": {
                "integration": "google-genai",
            },
        },
    )

    # Make a traced Gemini call
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain quantum computing in simple terms.",
    )

    print(response.text)


if __name__ == "__main__":
    main()