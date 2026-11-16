"""Role-based multi-agent workflow with optional OpenAI integration."""
import os
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Enterprise Multi-Agent AI", version="1.0.0")

class RunRequest(BaseModel):
    task: str = Field(min_length=5, max_length=4000)

class AgentResult(BaseModel):
    agent: str
    output: str

def ask_model(role: str, task: str, context: str = "") -> str:
    """Call OpenAI when configured; otherwise return a useful demo result."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return f"[Demo mode] {role} reviewed the task: {task}. Context: {context or 'No prior context.'}"
    from openai import OpenAI
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": f"You are the {role} agent. Be concise, factual, and identify assumptions."},
            {"role": "user", "content": f"Task: {task}\nPrevious agent output: {context}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or "No response."

def run_workflow(task: str) -> List[AgentResult]:
    plan = ask_model("Planning", task)
    research = ask_model("Research", task, plan)
    analysis = ask_model("Analysis", task, research)
    review = ask_model("Review", task, analysis)
    return [
        AgentResult(agent="planner", output=plan),
        AgentResult(agent="researcher", output=research),
        AgentResult(agent="analyst", output=analysis),
        AgentResult(agent="reviewer", output=review),
    ]

@app.get("/health")
def health():
    return {"status": "ok", "llm_enabled": bool(os.getenv("OPENAI_API_KEY"))}

@app.post("/run")
def run(request: RunRequest):
    results = run_workflow(request.task)
    return {
        "task": request.task,
        "agents": [item.model_dump() for item in results],
        "final_report": results[-1].output,
        "human_review_recommended": any(word in request.task.lower() for word in ("medical", "legal", "investment", "safety")),
    }
