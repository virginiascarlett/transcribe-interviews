#!/usr/bin/env python
import os
from openai import OpenAI, AuthenticationError, RateLimitError, BadRequestError
from dotenv import load_dotenv

# Get env variables
load_dotenv()
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_API_BASE = os.getenv("OPENAI_API_BASE")
LLM_TEST_MODEL = os.getenv("OPENAI_API_TEST_MODEL")

def query_LLM(instructions: str, user_data: str, model: str = "claude-v4.6-sonnet") -> str:
    """
    Sends a structured prompt to the OpenAI (or compatible) API.

    Prerequisites:
    Set these environment variables in your system or virtual environment:
    - LLM_API_BASE (e.g., https://<your-api-id>.execute-api.us-east-1.amazonaws.com/v1)
    - LLM_API_KEY (Your provisioned API key)
    - LLM_TEST_MODEL, LLM_PROD_MODEL (The models you want to use for testing and production, respectively)
    """

    # 1. Initialize the client securely using environment variables
    # We override the default OpenAI base URL with our URL
    client = OpenAI(
        base_url=LLM_API_BASE,
        api_key=LLM_API_KEY,
    )

    try:
        # 2. Make the request to the /chat/completions endpoint
        response = client.chat.completions.create(
            model=model,
            messages=[
                # The 'system' role is for overarching instructions
                {"role": "system", "content": instructions},
                # The 'user' role contains the specific data or query to process
                {"role": "user", "content": user_data}
            ],
            temperature=0.7 # Optional: 0.0 is very strict/analytical, 1.0+ is more creative
        )

        # 3. Extract and return just the text content from the AI's response
        return response.choices[0].message.content

    # 4. Handle potential API errors gracefully
    except AuthenticationError:
        return "Error: Invalid API key. Please check your credentials."
    except RateLimitError:
        return "Error: You have exceeded your monthly token quota."
    except BadRequestError as e:
        return f"Error: Bad request made to the server. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# ---------------------------------------------------------
# Test Execution Block
# Only runs if you execute this file directly
# ---------------------------------------------------------
if __name__ == "__main__":
    print("--- Testing Provider Module: OpenAI ---")

    # 1. Verify environment variables are present
    base_url = os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not base_url or not api_key:
        print("TEST FAILED: Missing Credentials.")
        print("Please set OPENAI_API_KEY and OPENAI_API_BASE in your environment.")
        exit(1)

    # 2. Define test parameters
    INSTRUCTIONS = "You are a helpful IT assistant. Answer in exactly one sentence."
    USER_DATA = "What is the main benefit of using a REST API?"

    print(f"Targeting URL: {base_url}")
    print(f"Sending request: {USER_DATA}\n")

    # 3. Call the abstracted function
    result = query_LLM(
        instructions=INSTRUCTIONS,
        user_data=USER_DATA,
        model=LLM_TEST_MODEL
    )

    # 4. Display the results
    print("--- RESPONSE ---")
    print(result)
    print("----------------")