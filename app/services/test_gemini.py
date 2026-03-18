import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# model = genai.GenerativeModel("gemini-pro")

prompt = """
Explain Early Blight in tomatoes.
Include causes, treatment and prevention.
Keep it simple.
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt
)

print(response.text)
