# app.py
"""
Streamlit app that uses LangGraph for workflow orchestration + Proctoring.
Run: streamlit run app.py
"""

import os
import sys
import subprocess
import time
import requests
import threading
import json
import pandas as pd
from typing import TypedDict, Any, Dict
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from langgraph.graph import StateGraph, START, END

from evaluator import (
    QUESTION_BANK,
    cached_llm_grade,
    rule_based_score,
    semantic_score,
    llm_grade,
    needs_followup,
    generate_followup,
)

# ------------------ Streamlit Setup ------------------
st.set_page_config(page_title="AI Interviewer — LangGraph + Gemini", layout="centered")
st.title("AI Interviewer — LangGraph + Gemini (PoC)")

# ------------------ Proctoring ------------------
def get_violation_count(candidate: str) -> int:
    """Poll the proctor server for number of violations logged for candidate."""
    try:
        port = int(os.getenv("PROCTOR_PORT", "8765"))
        base_url = os.getenv("PROCTOR_URL", f"http://127.0.0.1:{port}")
        resp = requests.get(f"{base_url}/violations?candidate={candidate}", timeout=1)
        if resp.status_code == 200:
            data = resp.json()
            return len(data.get("violations", []))
    except Exception:
        return 0
    return 0

def start_proctor_server_if_needed():
    """Start FastAPI proctor server (proctor_api.py) in background if not running."""
    port = int(os.getenv("PROCTOR_PORT", "8765"))
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/docs", timeout=1)
        if resp.status_code:
            return
    except Exception:
        pass

    proctor_script = os.path.join(os.path.dirname(__file__), "proctor_api.py")
    if not os.path.exists(proctor_script):
        st.warning("proctor_api.py not found — proctoring will not run.")
        return

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "proctor_api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)


def render_proctor_component(candidate_name: str, snapshot_interval_sec: int = 15):
    """Inject JS to enable webcam, detect cheating, block copy/paste, etc."""
    proctor_host = f"http://127.0.0.1:{os.getenv('PROCTOR_PORT','8765')}"
    html = f"""
    <div style="border:1px solid #f0f0f0;padding:8px;border-radius:6px;margin-bottom:8px;">
      <div id="proctor-status">Proctoring: requesting camera...</div>
      <video id="proctor-video" autoplay playsinline style="width:240px;height:180px;background:#000;"></video>
      <div id="proctor-warning" style="color:red;margin-top:6px;"></div>
    </div>
    <script>
    (function() {{
      const candidate = "{candidate_name or 'candidate'}";
      const snapshotInterval = {snapshot_interval_sec} * 1000;
      const proctorHost = "{proctor_host}";
      let vc = document.createElement('canvas');
      vc.width = 640; vc.height = 480;
      let stream = null;

      async function postViolation(event, details="") {{
        try {{
          await fetch(proctorHost + "/violation", {{
            method: "POST",
            body: new URLSearchParams({{candidate, timestamp: new Date().toISOString(), event, details}})
          }});
        }} catch(e) {{ console.warn("Violation post failed", e); }}
      }}

      async function postSnapshot(b64) {{
        try {{
          await fetch(proctorHost + "/snapshot", {{
            method: "POST",
            body: new URLSearchParams({{candidate, timestamp: new Date().toISOString(), image_b64: b64}})
          }});
        }} catch(e) {{ console.warn("Snapshot post failed", e); }}
      }}

      async function startCamera() {{
        try {{
          const v = document.getElementById("proctor-video");
          stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
          v.srcObject = stream;
          document.getElementById("proctor-status").innerText = "Proctoring: camera active";
          setInterval(() => {{
            try {{
              if (!stream) return;
              const ctx = vc.getContext('2d');
              ctx.drawImage(v, 0, 0, vc.width, vc.height);
              const dataUrl = vc.toDataURL('image/jpeg', 0.7);
              postSnapshot(dataUrl.split(',')[1]);
            }} catch(err) {{ console.warn("capture err", err); }}
          }}, snapshotInterval);
        }} catch(e) {{
          document.getElementById("proctor-warning").innerText = "Camera permission required — violation logged.";
          postViolation("camera_denied", String(e));
        }}
      }}

      document.addEventListener("visibilitychange", () => {{
        if (document.visibilityState !== "visible") {{
          document.getElementById("proctor-warning").innerText = "Tab switched — violation logged.";
          postViolation("tab_hidden", document.visibilityState);
        }}
      }});
      window.addEventListener("blur", () => postViolation("window_blur",""));
      document.addEventListener("paste", (e) => {{
        if (["TEXTAREA","INPUT"].includes(e.target.tagName)) {{
          e.preventDefault();
          postViolation("paste_attempt","");
          document.getElementById("proctor-warning").innerText = "Paste blocked.";
        }}
      }});
      document.addEventListener("copy", (e)=>{{ e.preventDefault(); postViolation("copy_attempt",""); }});
      document.addEventListener("cut", (e)=>{{ e.preventDefault(); postViolation("cut_attempt",""); }});

      try {{
        const bc = new BroadcastChannel('proctor_channel');
        bc.postMessage({{type:'hello',ts:Date.now()}});
        bc.onmessage = (ev)=>{{ if(ev.data.type==='hello') postViolation("multiple_tabs","detected"); }};
      }} catch(e) {{ console.warn("BC unavailable", e); }}

      startCamera();
    }})();
    </script>
    """
    components.html(html, height=260, scrolling=False)


# ------------------ Interview State ------------------
class InterviewState(TypedDict):
    q_index: int
    transcript: list
    followup_queue: list
    metadata: dict
    finished: bool
    current_question: dict | None


# Build LangGraph
builder = StateGraph(InterviewState)

# ------------------ Nodes ------------------
def ask_question_node(state: InterviewState) -> dict:
    cand = state.get("metadata", {}).get("name", "candidate")
    vcount = get_violation_count(cand)
    if vcount >= 2:
        st.error("Interview ended due to repeated proctoring violations (2 strikes).")
        return {"finished": True}
    if state.get("finished"):
        return {"finished": True}
    if state.get("current_question"):
        return {}

    q_index = state.get("q_index", 0)
    max_q = state.get("metadata", {}).get("max_questions", len(QUESTION_BANK))

    if q_index < min(max_q, len(QUESTION_BANK)):
        q = QUESTION_BANK[q_index].copy()
        q["is_followup"] = False
        return {"current_question": q}
    elif state.get("followup_queue"):
        fu = state["followup_queue"].pop(0)
        q = {
            "id": f"followup_{len(state.get('transcript', [])) + 1}",
            "title": "Follow-up",
            "prompt": fu,
            "rubric": {"points": 2, "criteria": [{"k":"clarify","weight":1,"desc":"Clarify"}]},
            "is_followup": True,
        }
        return {"current_question": q, "followup_queue": state["followup_queue"]}
    else:
        return {"finished": True}


def collect_answer_node(state: InterviewState) -> dict:
    q = state.get("current_question")
    if not q:
        return {}

    st.subheader(f"Q: {q.get('title')}")
    st.write(q.get("prompt"))

    answer_key = f"ans_{state.get('q_index',0)}_{q.get('id')}"
    ans_val = st.text_area(
        "Your answer",
        value=st.session_state.get(answer_key, ""),
        key=answer_key,
        height=180
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    submitted = False
    action = None

    if c1.button("Submit Answer", key=f"submit_{answer_key}"):
        submitted = True
        action = "submit"
    if c2.button("Skip", key=f"skip_{answer_key}"):
        submitted = True
        action = "skip"
    if c3.button("End Interview", key=f"end_{answer_key}"):
        return {"finished": True}

    if not submitted:
        st.stop()

    answer_text = ans_val.strip() if action == "submit" else "[SKIPPED]"

    # --- Scoring with spinner ---
    with st.spinner("Scoring your answer..."):
        rule = rule_based_score(answer_text, q.get("rubric", {"points": 0, "criteria": []}))
        keywords = [c["k"] for c in q.get("rubric", {}).get("criteria", [])]
        sem = semantic_score(answer_text, keywords)

    # --- LLM grading with spinner + cache ---
    with st.spinner("Getting AI feedback..."):
        llmfb = cached_llm_grade(answer_text, q.get("prompt", ""))

    record = {
        "q_id": q.get("id"),
        "question": q.get("prompt"),
        "answer": answer_text,
        "rule_score": rule,
        "semantic_score": sem,
        "llm_feedback": llmfb,
        "is_followup": q.get("is_followup", False),
        "timestamp": datetime.now().isoformat()
    }

    transcript = list(state.get("transcript", [])) + [record]

    # --- Follow-up decision ---
    followup_queue = list(state.get("followup_queue", []))
    if (not q.get("is_followup")) and needs_followup(
        rule_score=rule,
        answer=answer_text,
        rubric=q.get("rubric", {})
    ):
        fu = generate_followup(answer_text, q.get("prompt", ""))
        if fu:
            followup_queue.append(fu)

    q_index = state.get("q_index", 0)
    if not q.get("is_followup"):
        q_index += 1

    new_state = {
        "transcript": transcript,
        "followup_queue": followup_queue,
        "q_index": q_index,
        "current_question": None,
    }

    if "graph_state" in st.session_state:
        st.session_state["graph_state"].update(new_state)

    return new_state


def end_interview_node(state: InterviewState) -> dict:
    st.success("Interview finished — generating report")
    transcript = state.get("transcript", [])
    if not transcript:
        st.write("No answers recorded.")
        return {}

    df = pd.DataFrame([
        {
            "q_id": r["q_id"],
            "question": r["question"],
            "answer": r["answer"],
            "points": r["rule_score"]["points"],
            "max_points": r["rule_score"]["max_points"],
            "semantic": round(r.get("semantic_score", 0.0), 2),
        }
        for r in transcript
    ])
    st.dataframe(df)

    total = sum(r["rule_score"]["points"] for r in transcript)
    total_max = sum(r["rule_score"]["max_points"] for r in transcript)
    st.write(f"Total Score (rule-based): {total} / {total_max}")

    if st.button("Export JSON"):
        report = {
            "meta": state.get("metadata", {}),
            "transcript": transcript,
            "total": total,
            "total_max": total_max,
            "generated_at": datetime.now().isoformat(),
        }
        st.download_button("Download JSON", json.dumps(report, indent=2), file_name=f"report_{state.get('metadata',{}).get('name','candidate')}.json")

    return {}

# ------------------ Graph Build ------------------
builder.add_node("ask_question", ask_question_node)
builder.add_node("collect_answer", collect_answer_node)
builder.add_node("end_interview", end_interview_node)

builder.add_edge(START, "ask_question")
builder.add_edge("ask_question", "collect_answer")
builder.add_conditional_edges("collect_answer",
    lambda s: "end_interview" if s.get("finished") else "ask_question",
    {"end_interview":"end_interview","ask_question":"ask_question"})
builder.add_edge("end_interview", END)

app_graph = builder.compile()

# ------------------ UI ------------------
start_proctor_server_if_needed()

if "graph_state" not in st.session_state:
    with st.form("candidate_form"):
        st.header("Candidate details")
        name = st.text_input("Candidate name")
        role = st.selectbox("Role", ["Data Analyst", "Business Analyst", "Finance Analyst"])
        level = st.selectbox("Experience level", ["Junior", "Mid", "Senior"])
        total_qs = int(os.getenv("MAX_QUESTIONS", "10"))
        st.info(f"Total Questions this interview: {total_qs}")
        start = st.form_submit_button("Start Interview")
        if start:
            st.session_state["graph_state"] = {
                "q_index": 0,
                "transcript": [],
                "followup_queue": [],
                "metadata": {"name": name, "role": role, "level": level,
                             "started_at": datetime.now().isoformat(),
                             "max_questions": total_qs, "violations": 0},
                "finished": False,
                "current_question": None,
            }
            st.rerun()
    st.stop()

# Render proctoring UI once interview begins
render_proctor_component(st.session_state["graph_state"]["metadata"]["name"])

try:
    app_graph.invoke(st.session_state["graph_state"])
    st.session_state["graph_state"]["finished"] = True
except Exception as e:
    if "StreamlitAPIException" in str(type(e)):
        pass
    else:
        st.error(f"Graph execution error: {e}")
