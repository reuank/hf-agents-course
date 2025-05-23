from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, WebSearchTool, trace
import requests
from pydantic import BaseModel
load_dotenv()


class ResponseFormat(BaseModel):
    reasoning: str
    final_answer: str

agent = Agent(
    name="Assistant",
    # System prompt from here: https://huggingface.co/spaces/gaia-benchmark/leaderboard (minor adaptations)
    instructions="You are a general AI assistant. I will ask you a question. Report your reasoning, "
                 "and finish with your final answer. Your final answer should be a number OR as few words as possible "
                 "OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma "
                 "to write your number neither use units such as $ or percent sign unless specified otherwise. "
                 "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), "
                 "and write the digits in plain text unless specified otherwise. "
                 "If you are asked for a comma separated list, apply the above rules depending of whether the element "
                 "to be put in the list is a number or a string.",
    model="gpt-4o",
    tools=[WebSearchTool()],
    output_type=ResponseFormat
)

def submit(answers: list[dict[str, str]]):
    submission = {
        "username": "reuank",
        "agent_code": "https://github.com/reuank/hf-agents-course",
        "answer": answers
    }

    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    }

    result = requests.post("https://agents-course-unit4-scoring.hf.space/submit", headers=headers, json=submission)

    print(result.json())


async def main():
    questions: list[dict[str, str]] = requests.get('https://agents-course-unit4-scoring.hf.space/questions').json()
    answers: list[dict[str, str]] = []

    with trace(workflow_name="HuggingFace Agents Course"):
        for question in questions:
            print(f"Question: {question['question']}")

            if question["file_name"]:
                continue

            result = await Runner.run(agent, question["question"])
            answers.append(dict(task_id=question["task_id"], submitted_answer=result.final_output.final_answer))

    submit(answers)

if __name__ == "__main__":
    asyncio.run(main())