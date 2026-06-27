import json
import io
from .llm import chat

plans = {}


def extract_file_text(filename, content_bytes):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    text = ""
    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext in ("doc", "docx"):
        from docx import Document

        doc = Document(io.BytesIO(content_bytes))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        text = content_bytes.decode("utf-8", errors="replace")
    return text.strip()


def create_plan(topic, goals, duration_weeks=12, level="beginner", file_text=None):
    goals_text = goals
    if file_text:
        goals_text = f"{goals}\n\n--- File contents ---\n{file_text[:2000]}"

    prompt = (
        f"Create a {duration_weeks}-week learning plan for '{topic}'.\n"
        f"Student level: {level}\n"
        f"Goals and details: {goals_text}\n\n"
        "Return ONLY valid JSON with this structure:\n"
        '{"title": "...", "summary": "...", "weeks": [{"week": 1, "topic": "...", '
        '"objectives": ["..."], "exercises": ["..."]}]}'
    )
    result = chat(
        [{"role": "user", "content": prompt}],
        system_prompt_key="planner",
        temperature=0.7,
    )
    plan = _parse_plan(result)
    if plan:
        plan_id = str(len(plans) + 1)
        plan["id"] = plan_id
        plan["topic"] = topic
        plan["goals"] = goals
        plan["level"] = level
        plan["current_week"] = 1
        plans[plan_id] = plan
        return plan
    return None


def _parse_plan(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("\n", 1)[0]
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


def get_plan(plan_id):
    return plans.get(plan_id)


def list_plans():
    return [
        {"id": pid, "title": p.get("title", p["topic"])} for pid, p in plans.items()
    ]
