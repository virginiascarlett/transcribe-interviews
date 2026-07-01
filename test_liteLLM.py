#!/usr/bin/env python
import os
from litellm import completion
from dotenv import load_dotenv

# Get env variables
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
DATA_SUBDIR = os.getenv("DATA_SUBDIR")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE")
LITELLM_TEST_MODEL = os.getenv("LITELLM_TEST_MODEL")

def test_litellm_connection():
    # Make sure your API key is set. You can set it in your terminal environment
    # or uncomment the line below to set it directly in the script for testing.
    # os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

    # Change this to whatever model you are actually trying to use

    print(f"Testing LiteLLM connection using model: {LITELLM_TEST_MODEL}...")

    try:
        response = completion(
            model=LITELLM_TEST_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful test assistant."},
                {"role": "user", "content": "Hello! This is a test. Please reply with the exact phrase: 'Connection successful!'"}
            ],
            api_key=LITELLM_API_KEY,
            api_base=LITELLM_API_BASE
        )

        answer = response.choices[0].message.content

        print("\nAI response:")
        print(answer)

    except Exception as e:
        print(f"\nError connecting to LiteLLM:")
        print(e)

test_litellm_connection()