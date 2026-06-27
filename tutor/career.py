import json
import re
from .llm import chat
from .planner import create_plan, extract_file_text

career_tracks = {}


def generate_questions(name, file_text=None):
    extra = ""
    if file_text:
        extra = f"\nAdditional context from uploaded file:\n{file_text[:1000]}"
    prompt = (
        f"A user wants to start a career track in '{name}'.{extra}\n\n"
        "Generate 3-5 prerequisite questions to assess their current knowledge "
        "and tailor the learning path. Return ONLY valid JSON array of strings:\n"
        '["Question 1?", "Question 2?", ...]'
    )
    result = chat(
        [{"role": "user", "content": prompt}],
        system_prompt_key="career",
        temperature=0.7,
    )
    questions = _extract_json_array(result)
    if not questions:
        questions = [
            f"What is your current experience level with {name}?",
            f"What specific areas of {name} interest you most?",
            "How many hours per week can you dedicate to learning?",
            "Do you have any related background or prerequisites?",
        ]
    return questions


def create_track(name, questions, answers, file_text=None):
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))
    extra = ""
    if file_text:
        extra = f"\nAdditional context from file:\n{file_text[:1000]}"
    prompt = (
        f"Based on this career track '{name}', generate a structured list of skills "
        f"to learn in logical order (foundational first, advanced later).\n\n"
        f"User's Q&A:\n{qa}{extra}\n\n"
        "Return ONLY valid JSON array of objects:\n"
        '[{"name": "Skill Name", "description": "brief description"}, ...]\n'
        "Generate 3-8 skills."
    )
    result = chat(
        [{"role": "user", "content": prompt}],
        system_prompt_key="career",
        temperature=0.7,
    )
    skills = _extract_json_array(result)
    if not skills:
        skills = [
            {"name": f"{name} Fundamentals", "description": "Core concepts"},
            {"name": f"Intermediate {name}", "description": "Build on foundations"},
            {"name": f"Advanced {name}", "description": "Master the domain"},
        ]

    track_id = str(len(career_tracks) + 1)
    sub_tracks = []
    for i, skill in enumerate(skills):
        sub = {
            "id": str(i + 1),
            "name": skill["name"] if isinstance(skill, dict) else skill,
            "description": skill.get("description", "")
            if isinstance(skill, dict)
            else "",
            "completion": 0,
            "completed": False,
            "weeks": [],
        }
        sub_tracks.append(sub)

    career_tracks[track_id] = {
        "id": track_id,
        "name": name,
        "questions": questions,
        "answers": answers,
        "sub_tracks": sub_tracks,
    }
    return career_tracks[track_id]


def generate_sub_track_plan(track_id, sub_track_id):
    track = career_tracks.get(track_id)
    if not track:
        return None
    sub = next((s for s in track["sub_tracks"] if s["id"] == sub_track_id), None)
    if not sub:
        return None
    if sub["weeks"]:
        return sub

    goals = f"Learn {sub['name']} as part of {track['name']} career track"
    context = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in zip(track["questions"], track["answers"])
    )
    plan = create_plan(
        sub["name"],
        f"{goals}\n\nCareer context:\n{context}",
        duration_weeks=12,
        level="beginner",
        file_text=None,
    )
    if plan and "weeks" in plan:
        sub["weeks"] = plan["weeks"]
        sub["title"] = plan.get("title", sub["name"])
        sub["summary"] = plan.get("summary", "")
    else:
        sub["weeks"] = []
    return sub


def update_completion(track_id, sub_track_id, pct):
    track = career_tracks.get(track_id)
    if not track:
        return None
    sub = next((s for s in track["sub_tracks"] if s["id"] == sub_track_id), None)
    if not sub:
        return None
    sub["completion"] = min(max(pct, 0), 100)
    if sub["completion"] >= 100:
        sub["completed"] = True
    return sub


def get_track(track_id):
    return career_tracks.get(track_id)


def list_tracks():
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "sub_tracks": [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "completion": s["completion"],
                    "completed": s["completed"],
                }
                for s in t["sub_tracks"]
            ],
        }
        for t in career_tracks.values()
    ]


def _extract_json_array(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("\n", 1)[0]
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None
