from .llm import chat
from .planner import get_plan

sessions = {}


def create_session(plan_id):
    plan = get_plan(plan_id)
    if not plan:
        return None
    session_id = str(len(sessions) + 1)
    week = plan.get("current_week", 1)
    week_data = next((w for w in plan.get("weeks", []) if w["week"] == week), None)
    context = (
        f"Learning Plan: {plan['title']}\n"
        f"Topic: {plan['topic']}\n"
        f"Current Week {week}: {week_data['topic'] if week_data else 'General'}\n"
        f"Objectives: {', '.join(week_data['objectives']) if week_data else 'Master the topic'}\n"
    )
    sessions[session_id] = {
        "id": session_id,
        "plan_id": plan_id,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    f"Welcome to your tutoring session!\n\n"
                    f"We're covering **Week {week}**: {week_data['topic'] if week_data else 'Your topic'}.\n\n"
                    f"**Objectives:**\n"
                    + (
                        "\n".join(f"- {o}" for o in week_data["objectives"])
                        if week_data
                        else "- Master the material"
                    )
                    + "\n\nWhat would you like to learn about first?"
                ),
            }
        ],
        "context": context,
    }
    return sessions[session_id]


def send_message(session_id, user_message):
    session = sessions.get(session_id)
    if not session:
        return None
    session["messages"].append({"role": "user", "content": user_message})
    context_msg = {
        "role": "system",
        "content": f"Session context:\n{session['context']}",
    }
    full_messages = [context_msg] + session["messages"]
    reply = chat(full_messages, system_prompt_key="tutor", temperature=0.7)
    session["messages"].append({"role": "assistant", "content": reply})
    return {"reply": reply, "messages": session["messages"]}


def get_session(session_id):
    return sessions.get(session_id)


def list_sessions():
    return [
        {"id": sid, "plan_id": s["plan_id"], "message_count": len(s["messages"])}
        for sid, s in sessions.items()
    ]
