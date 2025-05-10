import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
from google import genai
from concurrent.futures import TimeoutError
from functools import partial

async def generate_with_timeout(client, prompt, timeout=10):
    """Generate content with a timeout"""
    print("Starting LLM generation...")
    try:
        # Convert the synchronous generate_content call to run in a thread
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        print("LLM generation completed")
        return response
    except TimeoutError:
        print("LLM generation timed out!")
        raise
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        raise

# Access your API key and initialize Gemini client correctly
api_key = "AIzaSyCtxeBmeKjjvhHhUblb_nSvm-fRtVHed7E"
client = genai.Client(api_key=api_key)

system_prompt = """
You are an AI assistant with full control over the user’s Windows environment. Perform the following steps exactly:

1. Open the Microsoft Paint application on this laptop.
2. Once Paint is open, select the “Rectangle” tool from the Shapes toolbar.
3. Draw a rectangle starting at coordinates (100, 100) and ending at coordinates (300, 200) on the blank canvas.
4. Ensure the rectangle has a solid black outline and no fill color.
5. Save the file as “rectangle.png” on the Desktop.
6. Close Microsoft Paint.

Confirm each step as you complete it.
"""


async def main():
    current_query = "Draw rectangle in a paint application"
    # Get model's response with timeout
    print("Preparing to generate LLM response...")
    prompt = f"{system_prompt}\n\nQuery: {current_query}"
    response = await generate_with_timeout(client, prompt)
    print(">>> response:" , response)


if __name__ == "__main__":
    asyncio.run(main())