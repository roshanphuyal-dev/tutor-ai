import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPTS = {
    "planner": (
        "You are an expert learning plan creator. Given a topic, goals, and duration, "
        "create a detailed, structured learning plan. Break it into weeks/modules with "
        "specific topics, resources, and exercises. Return the plan as a JSON object."
    ),
    "tutor": (
        "You are a knowledgeable, patient tutor helping a student learn. "
        "Adapt your teaching to their level. Use examples, analogies, and questions. "
        "Encourage active learning by asking the student questions. "
        "Be concise but thorough. If the student is stuck, try a different approach."
    ),
    "career": (
        "You are a career path expert and technical curriculum designer. "
        "Given a career track name and optional context, generate tailored "
        "prerequisite questions and a structured skill roadmap. "
        "Return responses as valid JSON."
    ),
}


def chat(messages, system_prompt_key="tutor", temperature=0.7):
    if not API_KEY:
        return _mock_reply(messages, system_prompt_key)

    import openai

    client_kwargs = {"api_key": API_KEY}
    if BASE_URL:
        client_kwargs["base_url"] = BASE_URL
    client = openai.OpenAI(**client_kwargs)
    system_content = SYSTEM_PROMPTS.get(system_prompt_key, SYSTEM_PROMPTS["tutor"])
    full_messages = [{"role": "system", "content": system_content}, *messages]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=full_messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except openai.AuthenticationError as e:
        hint = ""
        if not BASE_URL and not API_KEY.startswith("sk-"):
            hint = (
                " It looks like you're using a non-OpenAI API key. "
                "If you're using Google Gemini, set these in your .env file:\n"
                "  OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/\n"
                "  OPENAI_MODEL=gemini-2.0-flash"
            )
        return _mock_reply(messages, system_prompt_key, error_msg=str(e) + hint)
    except Exception as e:
        return _mock_reply(messages, system_prompt_key, error_msg=str(e))


def _mock_reply(messages, system_prompt_key, error_msg=None):
    last = messages[-1]["content"] if messages else ""
    err = f"\n\n> ⚠️ API Error: {error_msg}" if error_msg else ""

    if system_prompt_key == "planner":
        m = re.search(r"(\d+)-week", last)
        num_weeks = int(m.group(1)) if m else 4
        weeks = []
        for i in range(num_weeks):
            weeks.append(
                {
                    "week": i + 1,
                    "topic": f"Week {i + 1}: Topic Area",
                    "objectives": [
                        f"Learn key concepts for week {i + 1}",
                        "Complete hands-on exercises",
                    ],
                    "exercises": [
                        "Review material and take notes",
                        "Build a practice project",
                    ],
                }
            )
        return json.dumps(
            {
                "title": "Sample Learning Plan",
                "summary": f"Learning plan for: {last[:50]}...{err}",
                "weeks": weeks,
            }
        )
    return (
        "I'm your AI tutor! *(Running in offline/demo mode — "
        "check your API configuration)*\n\n"
        f'You asked: "{last[:200]}".'
        f"{err}"
    )
