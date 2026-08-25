#!/usr/bin/env python
import os
from litellm import completion
from dotenv import load_dotenv

# Get env variables
load_dotenv()
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE")
LITELLM_TEST_MODEL = os.getenv("LITELLM_TEST_MODEL")

def query_LLM(instructions: str, user_data: str, model: str = "gemini-3-flash-preview") -> str:
    """
    Sends a structured prompt to liteLLM.

    Prerequisites:
    Set these environment variables in your system or virtual environment:
    - LITELLM_API_BASE (https://litellm.dreamlab.ucsb.edu)
    - LITELLM_API_KEY (Your provisioned API key)
    - LITELLM_TEST_MODEL, LITELLM_PROD_MODEL (The models you want to use for testing and production, respectively)
    """

    try:
        response = completion(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_data},
        ],
        api_key=LITELLM_API_KEY,
        api_base=LITELLM_API_BASE
        )

        # Extract the text answer
        answer = response.choices[0].message.content
        return answer

    # 4. Handle potential API errors gracefully
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# ---------------------------------------------------------
# Test Execution Block
# Only runs if you execute this file directly
# ---------------------------------------------------------
if __name__ == "__main__":
    print("--- Testing LiteLLM Module ---")

    # 1. Verify environment variables are present
    base_url = os.environ.get("LITELLM_API_BASE")
    api_key = os.environ.get("LITELLM_API_KEY")

    if not base_url or not api_key:
        print("TEST FAILED: Missing Credentials.")
        print("Please set LITELLM_API_BASE and LITELLM_API_KEY in your environment.")
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
        model=LITELLM_TEST_MODEL
    )

    # 4. Display the results
    print("--- RESPONSE ---")
    print(result)
    print("----------------")