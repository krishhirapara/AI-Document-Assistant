import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key and api_key != "replace_this_with_your_actual_key":
    print("API key loaded successfully.")
else:
    print("API key not found. Check your .env file.")