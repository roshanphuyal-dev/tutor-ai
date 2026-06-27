import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tutor.planner import extract_file_text
from tutor.tutor_session import create_session, send_message, get_session, list_sessions
from tutor.career import (
    generate_questions,
    create_track,
    generate_sub_track_plan,
    update_completion,
    get_track,
    list_tracks,
)
from tutor.llm import RateLimitExceeded, get_rate_limit_status

APP_MODE = os.getenv("APP_MODE", "dev")

app = FastAPI(title="Tutor AI")

app.mount("/static", StaticFiles(directory="static", html=True), name="static")


class MessageRequest(BaseModel):
    message: str


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "retry_after": exc.retry_after},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) if str(exc) else "Internal server error"},
    )


@app.get("/api/config")
def get_config():
    return {
        "mode": APP_MODE,
        "rate_limit": get_rate_limit_status(),
    }


# --- Career Track Endpoints ---


@app.get("/api/career-tracks")
def get_career_tracks():
    return list_tracks()


@app.post("/api/career-tracks")
async def create_career_track(
    name: str = Form(...),
    file: UploadFile | None = File(None),
):
    file_text = None
    if file and file.filename:
        content = await file.read()
        if len(content) > 0:
            file_text = extract_file_text(file.filename, content)
    try:
        questions = generate_questions(name, file_text)
    except RateLimitExceeded:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"name": name, "questions": questions, "file_text": file_text}


@app.post("/api/career-tracks/{track_name}/finalize")
def finalize_career_track(
    track_name: str,
    questions: list[str] = Form(...),
    answers: list[str] = Form(...),
    file_text: str | None = Form(None),
):
    try:
        track = create_track(track_name, questions, answers, file_text)
    except RateLimitExceeded:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    if not track:
        raise HTTPException(500, "Failed to create career track")
    return track


@app.get("/api/career-tracks/{track_id}")
def get_career_track(track_id: str):
    track = get_track(track_id)
    if not track:
        raise HTTPException(404, "Career track not found")
    return track


@app.post("/api/career-tracks/{track_id}/sub-tracks/{sub_id}/generate")
def generate_sub_track(track_id: str, sub_id: str):
    sub = generate_sub_track_plan(track_id, sub_id)
    if not sub:
        raise HTTPException(404, "Sub-track not found")
    return sub


@app.post("/api/career-tracks/{track_id}/sub-tracks/{sub_id}/progress")
def update_sub_track_progress(track_id: str, sub_id: str, pct: int = Form(...)):
    sub = update_completion(track_id, sub_id, pct)
    if not sub:
        raise HTTPException(404, "Sub-track not found")
    return sub


# --- Session Endpoints ---


@app.get("/api/sessions")
def get_sessions():
    return list_sessions()


@app.post("/api/sessions")
def create_session_endpoint(plan_id: str):
    session = create_session(plan_id)
    if not session:
        raise HTTPException(404, "Plan not found")
    return session


@app.get("/api/sessions/{session_id}")
def get_session_endpoint(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@app.post("/api/sessions/{session_id}/chat")
def chat_endpoint(session_id: str, req: MessageRequest):
    try:
        result = send_message(session_id, req.message)
    except RateLimitExceeded:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    if not result:
        raise HTTPException(404, "Session not found")
    return result
