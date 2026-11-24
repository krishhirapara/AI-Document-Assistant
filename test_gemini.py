print("1. Starting test...", flush=True)

import os

from dotenv import load_dotenv
from google import genai

print("2. Packages imported.", flush=True)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key not found. Check your .env file.", flush=True)
else:
    print("3. API key loaded.", flush=True)

    client = None

    try:
        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": 90000},
        )

        print("4. Sending request to Gemini...", flush=True)

        interaction = client.interactions.create(
            model="gemini-3.5-flash-lite",
            input="Reply with only: Connection successful",
        )

        print("5. Response received.", flush=True)
        print(interaction.output_text or "No text returned.", flush=True)

    except Exception as error:
        print("Request failed.", flush=True)
        print(f"Error type: {type(error).__name__}", flush=True)
        print(str(error), flush=True)

    finally:
        if client is not None:
            client.close()