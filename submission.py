import os
import uuid

from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, WebSearchTool, trace, TResponseInputItem
import requests
from pydantic import BaseModel
load_dotenv()


COURSE_BASE_URL = "https://agents-course-unit4-scoring.hf.space"
GAIA_DATA_BASE_URL = "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation"

class ResponseFormat(BaseModel):
    reasoning: str
    final_answer: str

agent = Agent(
    name="GAIA Assistant",
    # System prompt from here: https://huggingface.co/spaces/gaia-benchmark/leaderboard (minor adaptations)
    instructions="You are a general AI assistant. I will ask you a question. Report your reasoning, "
                 "and finish with your final answer. Your final answer should be a number OR as few words as possible "
                 "OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma "
                 "to write your number neither use units such as $ or percent sign unless specified otherwise. "
                 "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), "
                 "and write the digits in plain text unless specified otherwise. "
                 "If you are asked for a comma separated list, apply the above rules depending of whether the element "
                 "to be put in the list is a number or a string. Use the tools available to you, especially web search.",
    model="gpt-4.1",
    tools=[WebSearchTool()],
    output_type=ResponseFormat
)

def submit(answers: list[dict[str, str]]):
    submission = {
        "username": "reuank",
        "agent_code": "https://github.com/reuank/hf-agents-course",
        "answers": answers
    }

    return requests.post(f"{COURSE_BASE_URL}/submit", json=submission).json()


async def main():
    conversation: list[TResponseInputItem] = []
    questions: list[dict[str, str]] = requests.get(f"{COURSE_BASE_URL}/questions").json()
    all_answers: list[dict[str, str]] = []
    group_id = uuid.uuid4().hex

    #with trace(workflow_name="HuggingFace Agents Course"):
    for question_id, question in enumerate(questions):
        with trace(workflow_name=f"Question {question['task_id']}", group_id=group_id):
            print("\n" + "="*50 + "\n")
            print(f"Answering question {question_id}: {question['question']}")
            print(f"ID: {question['task_id']}")

            if "youtube.com" in question['question']:
                print("Skipping this question, because it contains a video!")
                continue

            correct = False
            try_number = 1

            input_item = {
                "role": "user",
                "content": [{"type": "input_text", "text": question['question']}]
            }

            if question["file_name"]:
                file_url = f"{GAIA_DATA_BASE_URL}/{question['file_name']}"
                file = requests.get(file_url, headers={"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}, allow_redirects=True)
                file_name = file.headers["content-disposition"].split('filename="')[1].rstrip('";')
                file_extension = file_name.split(".")[-1]

                print(f"This question has a file! Processing {file_name}")

                if file_extension == "png":
                    input_item["content"].append({
                        "type": "input_image",
                        "image_url": file.url
                    })
                    print(f"Attached png to history")
                else:
                    print(f"Extension {file_extension} not supported")
                    continue

            conversation.append(input_item)

            while correct is False and try_number <= 5:
                llm_result = await Runner.run(agent, conversation)
                llm_answer = llm_result.final_output.final_answer
                conversation = llm_result.to_input_list()

                submission_result = submit([dict(task_id=question["task_id"], submitted_answer=llm_answer)])["score"]

                print(f"Try number {try_number} – LLM answered: {llm_answer} ({submission_result > 0})")

                if submission_result > 0:
                    correct = True
                    conversation.append({"content": f"Perfect, thank you!", "role": "user"})
                    all_answers.append(dict(task_id=question["task_id"], submitted_answer=llm_answer))
                else:
                    conversation.append({"content": f"No, that was wrong. Do not answer with {llm_answer} again. Try again!", "role": "user"})

                try_number += 1
                # print(json.dumps(conversation, indent=4))

        print(submit(all_answers))

if __name__ == "__main__":
    asyncio.run(main())