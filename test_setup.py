"""
Quick test script to verify Groq (text) and Gemini (vision) API keys are working.
Run this once after setting up .env — delete or ignore after confirming setup works.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# --- Test Groq (text model) ---
def test_groq():
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="openai/gpt-oss-20b",  # confirmed available via client.models.list()
        api_key=os.getenv("GROQ_API_KEY"),
    )
    response = llm.invoke("Say 'Groq is working!' in 5 words or less.")
    print("✅ GROQ RESPONSE:", response.content)


# --- Test Gemini (vision model) ---
def test_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    response = llm.invoke("Say 'Gemini is working!' in 5 words or less.")
    print("✅ GEMINI RESPONSE:", response.content)


if __name__ == "__main__":
    print("Testing Groq...")
    test_groq()

    print("\nTesting Gemini...")
    test_gemini()

    print("\n🎉 If both responses printed above, your setup is ready!")