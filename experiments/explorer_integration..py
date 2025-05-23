from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
from agents import Agent, Runner, function_tool

load_dotenv()

@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


external_client = AsyncOpenAI(
    base_url="https://explorer.invariantlabs.ai/api/v1/gateway/hf-agents-course/openai",
    default_headers={
        "Invariant-Authorization": "Bearer " + os.getenv("INVARIANT_API_KEY"),
    },
)

agent = Agent(
    name="Assistant",
    instructions="You only respond in haikus.",
    model=OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=external_client),
    tools=[get_weather],
)


async def main():
    result = await Runner.run(agent, "What's the weather in Tokyo?")

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())