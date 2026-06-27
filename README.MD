# To run it

cd tutor-ai
cp .env.example .env # Add your OpenAI API key here
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000/static/ in your browser

Features:
- Create learning plans (Topic, goals, duration, level)
- View weekly breakdown with objectives and exercises
- Start interactive tutoring sesions tied to your plan
- Chat with the AI tutor within the context of your current week

It works in mock/demo mode without an API key - but to get real AI responses, add your key to the.env