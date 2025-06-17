import os
import json
import httpx
import asyncio
import ipaddress
from openai import OpenAI
from rich import print as rich_print
from dotenv import load_dotenv

load_dotenv()

print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))
print("OPENROUTER_API_KEY:", os.environ.get("OPENROUTER_API_KEY"))

client = OpenAI(
    # api_key=os.environ.get("OPENROUTER_API_KEY"),
    # base_url="https://openrouter.ai/api/v1",
)


async def streamable_llm(messages, tools=[]):
    # Chat Completions API
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        # model="anthropic/claude-sonnet-4",
        # model="google/gemini-2.0-flash-001",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool["function"].__name__,
                    "parameters": tool["schema"],
                },
            }
            for tool in tools
        ],
        stream=True,
    )

    content = ""
    all_tool_calls = {}
    finish_reason = None

    for chunk in stream:
        # rich_print(chunk)
        token = chunk.choices[0].delta.content
        tool_calls = chunk.choices[0].delta.tool_calls
        if chunk.choices[0].finish_reason is not None:
            finish_reason = chunk.choices[0].finish_reason
        if token:
            content += token
            yield token
        elif tool_calls:
            for tool_call in tool_calls:
                if tool_call.index not in all_tool_calls:
                    all_tool_calls[tool_call.index] = {
                        "id": "",
                        "name": "",
                        "arguments": "",
                    }
                if tool_call.id:
                    all_tool_calls[tool_call.index]["id"] += tool_call.id
                if tool_call.function.name:
                    all_tool_calls[tool_call.index]["name"] += tool_call.function.name
                if tool_call.function.arguments:
                    all_tool_calls[tool_call.index][
                        "arguments"
                    ] += tool_call.function.arguments

    if all_tool_calls:
        rich_print(all_tool_calls)
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": tool_call["arguments"],
                        },
                    }
                    for tool_call in all_tool_calls.values()
                ],
            }
        )

    for tool_call in all_tool_calls.values():
        tool = next(
            (
                tool["function"]
                for tool in tools
                if tool["function"].__name__ == tool_call["name"]
            ),
            None,
        )

        if not tool:
            rich_print(f"Tool {tool_call["name"]} not found.")
            continue

        result = await tool(**json.loads(tool_call["arguments"]))

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result),
            }
        )
        rich_print("Function call result:", result)

    if finish_reason == "tool_calls":
        rich_print("Tool calls detected, continuing with the next iteration.")
        async for item in streamable_llm(messages):
            yield item


async def main():
    async def get_location_by_ip(ip: str) -> dict:
        """Get location information from an IP address string."""
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            raise ValueError(f"Invalid IP address: {ip}")

        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://ipleak.net/json/{ip}")
            response.raise_for_status()
            return response.json()

    tools = [
        {
            "function": get_location_by_ip,
            "schema": {
                "type": "object",
                "properties": {
                    "ip": {
                        "type": "string",
                        "anyOf": [{"format": "ipv4"}, {"format": "ipv6"}],
                    }
                },
                "required": ["ip"],
                "additionalProperties": False,
            },
        }
    ]

    stream = streamable_llm(
        [
            {
                "role": "user",
                "content": "What is the speed of light?"
                " and what is the location of ip 8.8.8.8 and 223.123.43.0?"
                " use tool for both ip addresses",
            }
        ],
        tools,
    )

    async for token in stream:
        rich_print(token, end="", flush=True)

    print("\n\nStream finished.")


asyncio.run(main())
