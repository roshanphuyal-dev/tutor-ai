import os
import json
import re
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_rate_limited_until = 0.0


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.0f}s")


def get_rate_limit_status():
    remaining = _rate_limited_until - time.time()
    return {"locked": remaining > 0, "retry_after_seconds": max(0, remaining)}


def _parse_retry_delay(error_body: dict) -> float:
    try:
        details = error_body.get("error", {}).get("details", [])
        for detail in details:
            if detail.get("@type", "").endswith("RetryInfo"):
                raw = detail.get("retryDelay", "0s")
                m = re.match(r"(\d+(?:\.\d+)?)\s*([smhd])", raw)
                if m:
                    v, unit = float(m.group(1)), m.group(2)
                    return v * {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit, 1)
    except Exception:
        pass
    return 60.0


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

    global _rate_limited_until
    now = time.time()
    if now < _rate_limited_until:
        raise RateLimitExceeded(_rate_limited_until - now)

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
    except openai.RateLimitError as e:
        retry_after = 60.0
        raw_body = getattr(e, "body", None)
        try:
            body = raw_body or (
                e.response.json() if hasattr(e, "response") and e.response else {}
            )
            if isinstance(body, list) and len(body) > 0:
                body = body[0]
            if isinstance(body, dict):
                retry_after = _parse_retry_delay(body)
        except Exception:
            pass

        if retry_after == 60.0 and hasattr(e, "message"):
            m = re.search(r"retry in ([\d.]+)s", str(e.message), re.I)
            if m:
                retry_after = float(m.group(1))

        _rate_limited_until = time.time() + max(retry_after, 10)
        raise RateLimitExceeded(retry_after) from e
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
    except openai.APIError as e:
        err_str = str(e)
        m = re.search(r"retry in ([\d.]+)s", err_str, re.I)
        if m:
            retry_after = float(m.group(1))
            _rate_limited_until = time.time() + retry_after
            raise RateLimitExceeded(retry_after) from e
        return _mock_reply(messages, system_prompt_key, error_msg=err_str)
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
