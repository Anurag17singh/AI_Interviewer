## AI Interviewer

## Design Document & Approach Strategy

* This is a short report explaining:

## Problem Statement: Need for an AI-driven interviewer with proctoring.

## System Design:

* LLM backends (Gemini primary, HuggingFace fallback, local transformer).

* LangGraph for state management (Ask → Answer → Evaluate → Follow-up → End).

* Rule-based scoring + Semantic similarity + LLM feedback.

* Proctoring server (camera on, tab monitoring, copy-paste blocking).

## Workflow:

* Candidate submits details.

* System asks main questions (limited by .env: MAX_QUESTIONS).

* Weak answers trigger follow-ups (via LLM).

* Proctor violations auto-end after 2 strikes.

* Transcript is saved/exported.

## Approach:

* Kept modular (evaluator.py, llm_model.py, app.py).

* Configurable environment via .env (HF_API_KEY, GEMINI_API_KEY, MAX_QUESTIONS).

* Caching (@st.cache_resource, @st.cache_data) for speed.

## Security:

* Copy/paste disabled in input field.

* Camera required (via browser).

* Tab switching tracked via WebSocket events.
