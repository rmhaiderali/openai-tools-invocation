import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))
print("OPENROUTER_API_KEY:", os.environ.get("OPENROUTER_API_KEY"))

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

parameters = {
    "type": "object",
    "properties": {
        "ip": {
            "type": "string",
            "anyOf": [{"format": "ipv4"}, {"format": "ipv6"}],
        }
    },
    "required": ["ip"],
    "additionalProperties": False,
}

# Chat Completions API
response = client.chat.completions.create(
    model="gpt-4o-mini",
    # model="anthropic/claude-sonnet-4",
    # model="google/gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "What is the location of ip 8.8.8.8?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_location_by_ip",
                "parameters": parameters,
            },
        }
    ],
)

print(response)
print(response.choices[0].message)
print(response.choices[0].message.tool_calls[0].function.name)
print(response.choices[0].message.tool_calls[0].function.arguments)


# Responses API
# response = client.responses.create(
#     model="gpt-4o-mini",
#     input=[{"role": "user", "content": "What is the location of ip 8.8.8.8?"}],
#     tools=[
#         {
#             "type": "function",
#             "name": "get_location_by_ip",
#             "parameters": parameters,
#         },
#     ],
# )

# print(response.output)
# print(response.output[0].name)
# print(response.output[0].arguments)
