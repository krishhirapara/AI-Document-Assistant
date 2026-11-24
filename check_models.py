import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key not found. Check your .env file.")
else:
    print("Connecting to Google's model-list endpoint...", flush=True)

    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 30000},
    )

    try:
        found_model = False

        for model in client.models.list():
            print(model.name, flush=True)
            found_model = True

        if found_model:
            print("\nModel listing successful!", flush=True)
        else:
            print("The request succeeded but returned no models.", flush=True)

    except Exception as error:
        print(f"\nError type: {type(error).__name__}", flush=True)
        print(str(error), flush=True)

    finally:
        client.close()