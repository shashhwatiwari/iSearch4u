from google import genai
import os
from dotenv import load_dotenv

load_dotenv(".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

#  1. The real Python function the model is allowed to trigger  / this gets replaced by an actual weather API in a real implementation.
def get_weather(city: str) -> str:
    fake_data = {
        "london": "12°C, overcast",
        "tokyo": "28°C, sunny",
        "new york": "18°C, partly cloudy",
    }
    return fake_data.get(city.lower(), f"No weather data for {city}")

#  2. The description of that function, written so the model can read it 
tools = [
    genai.types.Tool(
        function_declarations=[
            genai.types.FunctionDeclaration(
                name="get_weather",
                description="Returns the current weather for a given city.",
                parameters=genai.types.Schema(
                    type="OBJECT",
                    properties={
                        "city": genai.types.Schema(
                            type="STRING",
                            description="The name of the city"
                        )
                    },
                    required=["city"]
                )
            )
        ]
    )
]

#  3. The agent loop 
def run_agent(question: str):
    messages = [{"role": "user", "parts": [{"text": question}]}]

    while True:
        # Send the full conversation history to the model
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=genai.types.GenerateContentConfig(tools=tools)
        )

        part = response.candidates[0].content.parts[0]

        # Case A: the model wants to call a tool
        if part.function_call:
            fn_name = part.function_call.name   # "get_weather"
            fn_args = part.function_call.args   # {"city": "Tokyo"}

            # Run the actual Python function with the args the model chose
            result = get_weather(**fn_args)
            print(f"[Tool call] {fn_name}({fn_args}) → {result}")

            # Tell the model: "here is the tool call you made"
            messages.append({
                "role": "model",
                "parts": [{"function_call": {"name": fn_name, "args": fn_args}}]
            })

            # Tell the model: "here is what the tool returned"
            messages.append({
                "role": "user",
                "parts": [{
                    "function_response": {
                        "name": fn_name,
                        "response": {"result": result}
                    }
                }]
            })

            # Loop again — model will now write its final answer using the result

        # Case B: the model wrote a plain text answer — we're done
        elif part.text:
            print(f"\nFinal answer: {part.text}")
            break

if __name__ == "__main__":
    run_agent("What's the weather like in Tokyo?")