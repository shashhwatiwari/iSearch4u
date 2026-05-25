import types

from google import genai
import os
from dotenv import load_dotenv

load_dotenv(".env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

messages = []
while True:
    content = input("Ask a question or type 'quit' to exit: ")
    if content.lower() == "quit":
        break
    messages.append({"role": "user", "parts": [{"text": content}]}) # appending user messages in the schema that the Gemini API expects, different from OpenAI and Anthropic
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages
    )
    print(f"Response : {response.text}              Token used : {response.usage_metadata}")
    messages.append({"role": "model", "parts": [{"text": response.text}]})

print("Thanks for using the AI assistant!")
print("Conversation history:")
for msg in messages:
    print(f"{msg['role'].capitalize()}: {msg['parts'][0]['text']}")