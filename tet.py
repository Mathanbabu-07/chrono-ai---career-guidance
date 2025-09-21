# chrono_full_app.py
"""
Chrono — Career Guidance AI (full single-file Streamlit app)
Updated: Live HR mic capture + 25 automatic unique MCQs + improved UI and animations
"""

# ---------------------------
# Imports & optional libraries
# ---------------------------
import os
import sys
import time
import json
import html
import random
import base64
import re
import tempfile
import subprocess
import datetime
import wave
from io import BytesIO
from typing import List, Dict, Optional, Any

import streamlit as st
import streamlit.components.v1 as components

# Optional: generative client (google-generativeai)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Optional: pandas for nicer dataframes
try:
    import pandas as pd
except Exception:
    pd = None

# Optional: reportlab for PDF resume generation
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepInFrame
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
except Exception:
    SimpleDocTemplate = None

# Optional speech libraries
try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

try:
    import whisper
except Exception:
    whisper = None

# Optional webrtc (for live mic capture)
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
    import av
    import numpy as np
except Exception:
    webrtc_streamer = None
    av = None
    np = None

# ---------------------------
# Configuration - add API key here or in Settings
# ---------------------------
GEMINI_API_KEY = "AIzaSyBklQvZmFrtwpxuK516FZr-XVotKdR6CSU"  # <-- paste your API key here if desired (or in Settings runtime)
DEFAULT_MODEL_ID = ""  # optional; left blank by default

if genai is not None and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

# ---------------------------
# Streamlit page setup & CSS
# ---------------------------
st.set_page_config(page_title="Chrono — Career AI", page_icon="⏱️", layout="wide")

def inject_css(bg_data_url: Optional[str] = None):
    css = """
    <style>
    :root {
      --bg:#06070A; --muted:#9aa6b2; --accent:#9fe8ff; --accent2:#8cf5c6; --gold:#d4af37; --text:#eafcff;
    }
    html, body, .appview-container .main { background: transparent !important; }
    .stApp { background: transparent; }
    /* Animated background */
    .animated-bg {
      position: fixed; inset:0; z-index:-2;
      background: radial-gradient(600px 300px at 10% 20%, rgba(195,166,255,0.06), transparent 10%),
                  radial-gradient(500px 300px at 85% 80%, rgba(212,175,55,0.04), transparent 12%),
                  linear-gradient(180deg,#030412,#071018 40%, #0b1422 100%);
      animation: bgShift 20s linear infinite;
      opacity: 0.95;
    }
    @keyframes bgShift { 0%{filter:hue-rotate(0deg);}50%{filter:hue-rotate(18deg);}100%{filter:hue-rotate(0deg);} }
    .stApp::before { content:""; position:fixed; inset:0; z-index:-3; pointer-events:none; background-image: radial-gradient(rgba(255,255,255,0.03) 1px, transparent 1px); background-size: 80px 80px; opacity:0.06; }
    /* Top hero / cards */
    .top-hero { padding:18px; border-radius:12px; background:linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.02); }
    .brand { font-size:28px; font-weight:900; color:var(--accent); }
    .slogan { font-size:18px; font-weight:800; color:var(--text); }
    .muted { color:var(--muted); font-size:13px; margin-top:6px; }
    /* Sidebar style */
    .sidebar-card { padding:12px; border-radius:10px; background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.02)); border:1px solid rgba(255,255,255,0.02); margin-bottom:8px; }
    /* Chat panel + bubbles */
    .chat-panel { background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.04)); padding:12px; border-radius:12px; min-height:420px; max-height:72vh; overflow:auto; border:1px solid rgba(255,255,255,0.02); }
    .bubble-user { float:right; clear:both; background:linear-gradient(90deg,#0b2430,#071f28); color:var(--text); padding:12px; border-radius:12px; margin:10px 0; text-align:right; max-width:72%; box-shadow:0 8px 20px rgba(0,0,0,0.6); }
    .bubble-chrono { float:left; clear:both; background:linear-gradient(90deg,#063449,#0b3250); color:#fff; padding:12px; border-radius:12px; margin:10px 0; text-align:left; max-width:72%; box-shadow:0 8px 20px rgba(0,0,0,0.6); }
    .input-row { display:flex; gap:8px; align-items:center; margin-top:12px; }
    .center-input { display:flex; gap:8px; align-items:center; background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:10px 12px; border-radius:999px; border:1px solid rgba(255,255,255,0.03); flex:1; }
    .center-input input { background:transparent; border:0; outline:0; color:var(--text); width:100%; }
    .send-btn { width:46px; height:46px; border-radius:999px; border:none; background:linear-gradient(90deg,var(--accent2),var(--accent)); color:#021018; font-weight:800; cursor:pointer; }
    .tiny-loader { display:flex; gap:6px; align-items:center; justify-content:center; padding:6px; }
    .dot { width:8px; height:8px; border-radius:50%; background:linear-gradient(90deg,#6ee7ff,#c3a6ff); animation:beat 0.7s infinite; }
    .dot2 { animation-delay:0.09s;} .dot3 { animation-delay:0.18s;}
    @keyframes beat { 0%{transform:scale(.75);}50%{transform:scale(1.25);}100%{transform:scale(.75);} }
    .mcq-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-top:12px; }
    .mcq-btn { padding:12px; border-radius:10px; background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.03)); border:1px solid rgba(255,255,255,0.02); color:var(--text); cursor:pointer; }
    .mcq-btn:hover { transform: translateY(-4px); box-shadow: 0 14px 34px rgba(12,180,255,0.04); }
    .mcq-btn.selected { border:2px solid var(--accent); box-shadow:0 14px 40px rgba(195,166,255,0.06); }
    .table-area { background:linear-gradient(180deg,#021218,#031624); padding:12px; border-radius:8px; }
    .lottie-wrap { border-radius:8px; padding:8px; background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.03)); border:1px solid rgba(255,255,255,0.02); }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("<div class='animated-bg'></div>", unsafe_allow_html=True)
    if bg_data_url:
        overlay = f"<style> .stApp::before {{ background-image: url('{bg_data_url}'); background-size:cover; opacity:0.12; }} </style>"
        st.markdown(overlay, unsafe_allow_html=True)

# apply CSS (use previously uploaded background if present)
bg_data = st.session_state.get("custom_bg_dataurl", None)
inject_css(bg_data)

# ---------------------------
# Utility: parse JSON fallback
# ---------------------------
def parse_json_fallback(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if m:
        sub = m.group(1)
        try:
            return json.loads(sub)
        except Exception:
            try:
                return json.loads(sub.replace("'", '"'))
            except Exception:
                pass
    try:
        import ast
        cleaned = text.strip().strip('`')
        return ast.literal_eval(cleaned)
    except Exception:
        pass
    return None

# ---------------------------
# Utility: generate resume PDF
# ---------------------------
def generate_resume_pdf_bytes(resume_data: Dict[str, Any]) -> BytesIO:
    if SimpleDocTemplate is None:
        raise RuntimeError("reportlab not installed. pip install reportlab")
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    heading = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=11)
    small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#6b7280'))
    left = []
    left.append(Paragraph("<b>" + html.escape(resume_data.get("full_name","")) + "</b>", ParagraphStyle('Name', fontSize=16)))
    left.append(Paragraph(html.escape(resume_data.get("phone","")), small))
    left.append(Paragraph(html.escape(resume_data.get("email","")), small))
    if resume_data.get("linkedin"):
        left.append(Paragraph("LinkedIn: " + html.escape(resume_data.get("linkedin","")), small))
    left.append(Spacer(1,8))
    if resume_data.get("languages"):
        left.append(Paragraph("<b>Languages</b>", heading))
        for ln in resume_data.get("languages", []):
            left.append(Paragraph(html.escape(ln), normal))
        left.append(Spacer(1,6))
    if resume_data.get("certifications"):
        left.append(Paragraph("<b>Certifications</b>", heading))
        for c in resume_data.get("certifications", []):
            left.append(Paragraph(html.escape(c), normal))
        left.append(Spacer(1,6))
    if resume_data.get("address"):
        left.append(Paragraph("<b>Address</b>", heading))
        left.append(Paragraph(html.escape(resume_data.get("address","")), normal))
    right = []
    right.append(Paragraph("<b>Summary</b>", heading))
    right.append(Paragraph(html.escape(resume_data.get("objective","")), normal))
    right.append(Spacer(1,6))
    if resume_data.get("education"):
        right.append(Paragraph("<b>Education</b>", heading))
        for ed in resume_data.get("education", []):
            deg = html.escape(ed.get("degree",""))
            inst = html.escape(ed.get("institution",""))
            year = html.escape(ed.get("year",""))
            cgpa = ed.get("cgpa","")
            right.append(Paragraph(f"{deg} — {inst} ({year})", normal))
            if cgpa:
                right.append(Paragraph("CGPA: " + html.escape(str(cgpa)), small))
        right.append(Spacer(1,6))
    if resume_data.get("projects"):
        right.append(Paragraph("<b>Projects</b>", heading))
        for p in resume_data.get("projects", []):
            right.append(Paragraph("<b>" + html.escape(p.get("title","")) + "</b>", normal))
            if p.get("desc"):
                right.append(Paragraph(html.escape(p.get("desc")), small))
        right.append(Spacer(1,6))
    if resume_data.get("skills"):
        right.append(Paragraph("<b>Skills</b>", heading))
        right.append(Paragraph(", ".join([html.escape(s) for s in resume_data.get("skills", [])]), normal))
    left_k = KeepInFrame(160,700,left,mergeSpace=True)
    right_k = KeepInFrame(360,700,right,mergeSpace=True)
    table = Table([[left_k, right_k]], colWidths=[160,360])
    table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8)]))
    doc.build([table])
    buf.seek(0)
    return buf

# ---------------------------
# TTS & transcription helpers
# ---------------------------
def tts_bytes_gtts(text: str) -> Optional[bytes]:
    if gTTS is None:
        return None
    try:
        t = gTTS(text=text, lang='en')
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        t.save(tmp.name)
        with open(tmp.name, 'rb') as f:
            data = f.read()
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        return data
    except Exception:
        return None

def transcribe_audio_file(path: str) -> str:
    # prefer whisper if available
    if whisper is not None:
        try:
            model = whisper.load_model("small")
            res = model.transcribe(path)
            return res.get("text","")
        except Exception:
            pass
    if sr is not None:
        try:
            r = sr.Recognizer()
            with sr.AudioFile(path) as src:
                a = r.record(src)
            try:
                return r.recognize_google(a)
            except Exception:
                return ""
        except Exception:
            return ""
    return ""

# ---------------------------
# Generative query helper (auto-discovery)
# ---------------------------
def pick_working_model():
    if genai is None:
        return ""
    try:
        if hasattr(genai, "list_models"):
            models = genai.list_models()
            model_ids = []
            for m in models:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("id") or ""
                else:
                    name = getattr(m, "name", "") or getattr(m, "id", "")
                if name:
                    model_ids.append(name)
            for candidate in model_ids:
                low = candidate.lower()
                if "gemini" in low or "flash" in low or "bison" in low:
                    return candidate
            return model_ids[0] if model_ids else ""
    except Exception:
        return ""
    return ""

def query_generative(prompt: str, system_instruction: Optional[str] = None, model_id: Optional[str] = None, max_output_tokens: int = 512) -> str:
    if genai is None:
        return "ERROR: generative client not installed. Install google-generativeai."
    api_key = st.session_state.get("api_key","") or GEMINI_API_KEY or ""
    if not api_key:
        return "ERROR: API key not set. Paste it into Settings."
    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass
    full = prompt
    if system_instruction:
        full = "SYSTEM: " + system_instruction + "\n\nUSER: " + prompt
    mid = model_id or st.session_state.get("model_id","") or DEFAULT_MODEL_ID or ""
    try:
        if mid:
            m = genai.GenerativeModel(mid)
        else:
            m = genai.GenerativeModel()
        resp = m.generate_content(full)
        if hasattr(resp, "text"):
            return resp.text
        return str(resp)
    except Exception as e:
        err = str(e)
        if "not found" in err.lower() or "is not found" in err.lower() or "not supported" in err.lower() or "404" in err:
            candidate = pick_working_model()
            if candidate:
                try:
                    m2 = genai.GenerativeModel(candidate)
                    resp2 = m2.generate_content(full)
                    if hasattr(resp2, "text"):
                        st.session_state["model_id"] = candidate
                        return resp2.text
                    return str(resp2)
                except Exception as e2:
                    return "ERROR: model discovery failed: " + str(e2) + " | original: " + err
            else:
                return "ERROR: configured model not found. Paste API key in Settings."
        return "ERROR: " + err

# ---------------------------
# MCQ Banks (expand as needed) - unique ids included
# ---------------------------
APTITUDE_BANK: List[Dict[str, Any]] = [
    {"id":"apt_1","q":"5 people can do a job in 20 days. How many people to do the job in 5 days?","options":["20","25","5","10"],"answer":0,"explanation":"5*20=100 person-days; 100/5 = 20."},
    {"id":"apt_2","q":"A can do a task in 12 days and B in 18 days. How long together?","options":["6 days","7.2 days","9 days","10 days"],"answer":1,"explanation":"1/12+1/18 = 5/36 -> 36/5 = 7.2."},
    {"id":"apt_3","q":"If x + 1/x = 5, what is x^2 + 1/x^2?","options":["21","23","25","27"],"answer":1,"explanation":"(x+1/x)^2 = x^2 + 2 + 1/x^2 -> 25 - 2 = 23."},
    {"id":"apt_4","q":"Two trains 300 km apart travel towards each other at 60 km/h and 40 km/h. Time to meet?","options":["3 hours","5 hours","2.5 hours","4 hours"],"answer":0,"explanation":"Relative speed 100 km/h -> 300/100 = 3 hours."},
    {"id":"apt_5","q":"Average of five numbers is 20. Sum is?","options":["100","80","25","120"],"answer":0,"explanation":"Average * count = sum -> 20*5 = 100."},
    # Add many more items for production
]

TECH_MCQ_BANKS: Dict[str, List[Dict[str, Any]]] = {
    "python": [
        {"id":"py_1","q":"What is the output of: print(2**3**2)?","options":["64","512","Error","9"],"answer":1,"explanation":"3**2=9 -> 2**9=512 (right-associative)."},
        {"id":"py_2","q":"Which statement creates a generator?","options":["def f(): yield 1","def f(): return 1","lambda x: x","list()"],"answer":0,"explanation":"yield creates generator functions."},
        {"id":"py_3","q":"Which data type is immutable?","options":["list","dict","tuple","set"],"answer":2,"explanation":"tuple is immutable."},
        {"id":"py_4","q":"Which method appends to list end?","options":["append","push","add","extend"],"answer":0,"explanation":"append adds a single element to end."},
        {"id":"py_5","q":"Which builtin provides LRU cache?","options":["functools.lru_cache","cachetools","custom class","threading"],"answer":0,"explanation":"functools.lru_cache is builtin."},
    ],
    "c": [
        {"id":"c_1","q":"Which header contains printf?","options":["<stdio.h>","<stdlib.h>","<conio.h>","<io.h>"],"answer":0,"explanation":"printf declared in stdio.h."},
        {"id":"c_2","q":"Integer division 7/2 yields?","options":["3","3.5","4","Error"],"answer":0,"explanation":"Integer division truncates -> 3."},
    ],
    "cpp": [
        {"id":"cpp_1","q":"'virtual' keyword used for?","options":["Dynamic dispatch","Memory allocation","Input/Output","Templates"],"answer":0,"explanation":"virtual enables runtime polymorphism."},
        {"id":"cpp_2","q":"Feature specific to C++ not in C?","options":["Classes","Pointers","Macros","stdio"],"answer":0,"explanation":"Classes/OOP are a C++ feature."},
    ],
    "java": [
        {"id":"java_1","q":"Entry-point signature in Java?","options":["public static void main(String[] args)","void main()","static main()","public void main()"],"answer":0,"explanation":"Exact signature required."},
    ],
    "data structures": [
        {"id":"ds_1","q":"Binary search complexity?","options":["O(log n)","O(n)","O(n log n)","O(1)"],"answer":0,"explanation":"Binary search halves search each step."},
        {"id":"ds_2","q":"Which uses LIFO?","options":["Queue","Stack","Heap","Graph"],"answer":1,"explanation":"Stack is LIFO."},
    ],
    "database": [
        {"id":"db_1","q":"Which removes a table?","options":["DROP TABLE","DELETE TABLE","TRUNCATE ROW","REMOVE TABLE"],"answer":0,"explanation":"DROP TABLE removes structure & data."},
        {"id":"db_2","q":"Normalization reduces what?","options":["Redundancy","Security","Speed","Memory"],"answer":0,"explanation":"Reduce redundancy."},
    ],
    "os": [
        {"id":"os_1","q":"Which scheduling is preemptive?","options":["Round Robin","FCFS","SJF non-preemptive","None"],"answer":0,"explanation":"Round Robin is preemptive."}
    ],
    "web": [
        {"id":"web_1","q":"HTML tag for hyperlink?","options":["<a>","<link>","<href>","<url>"],"answer":0,"explanation":"<a href='...'> is used."},
        {"id":"web_2","q":"CSS property for background color?","options":["background-color","bg","color-bg","bcolor"],"answer":0,"explanation":"background-color sets background."}
    ]
}

# ---------------------------
# Unique sampling helpers (no repeats)
# ---------------------------
def sample_unique_apti(count: int) -> List[Dict[str, Any]]:
    used = st.session_state.setdefault("used_apt_ids", set())
    available = [q for q in APTITUDE_BANK if q["id"] not in used]
    picked = []
    if len(available) >= count:
        picked = random.sample(available, k=count)
    else:
        picked = available.copy()
        # reset used for bank if needed
        bank_ids = {q["id"] for q in APTITUDE_BANK}
        used.difference_update(bank_ids)
        pool = [q for q in APTITUDE_BANK if q["id"] not in used and q not in picked]
        while len(picked) < count and pool:
            cand = random.choice(pool)
            picked.append(cand)
            pool.remove(cand)
        while len(picked) < count:
            picked.append(random.choice(APTITUDE_BANK))
    for q in picked:
        used.add(q["id"])
    return picked

def sample_unique_multi_topic(topics: List[str], count: int) -> List[Dict[str, Any]]:
    # Build combined pool with topic prefix to ensure unique id across topics
    pool = []
    for t in topics:
        bank = TECH_MCQ_BANKS.get(t, [])
        for q in bank:
            item = q.copy()
            item["_topic"] = t
            item["_global_id"] = f"{t}:{q['id']}"
            pool.append(item)
    used = st.session_state.setdefault("used_tech_ids", set())
    available = [q for q in pool if q["_global_id"] not in used]
    picked = []
    if len(available) >= count:
        picked = random.sample(available, k=count)
    else:
        picked = available.copy()
        # clear used for these topics if pool exhausted
        topic_ids = {q["_global_id"] for q in pool}
        used.difference_update(topic_ids)
        pool2 = [q for q in pool if q["_global_id"] not in used and q not in picked]
        while len(picked) < count and pool2:
            cand = random.choice(pool2)
            picked.append(cand)
            pool2.remove(cand)
        while len(picked) < count:
            picked.append(random.choice(pool))
    for q in picked:
        used.add(q["_global_id"])
    # remove helper keys before return
    for q in picked:
        q.pop("_global_id", None)
        q.pop("_topic", None)
    return picked

# ---------------------------
# Session initialization
# ---------------------------
if "chrono_msgs" not in st.session_state:
    st.session_state["chrono_msgs"] = [{"role":"assistant","content":"Hello — I'm Chrono. Ask about career guidance, jobs, interviews, or resume help."}]
if "job_history" not in st.session_state:
    st.session_state["job_history"] = []
if "model_id" not in st.session_state:
    st.session_state["model_id"] = DEFAULT_MODEL_ID or ""
if "api_key" not in st.session_state:
    st.session_state["api_key"] = GEMINI_API_KEY or ""

# ---------------------------
# Sidebar navigation & branding
# ---------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-card' style='text-align:center'><div style='font-size:20px;color:#7ee9ff;font-weight:800'>Chrono</div><div class='muted'>Career Guidance AI</div></div>", unsafe_allow_html=True)
    page = st.radio("Features", ["Home", "Chrono AI", "Job Info", "Resume Builder", "Mock Interview", "Settings"], index=1)
    st.markdown("<div class='muted' style='margin-top:8px'>Tip: For live model responses paste API key in Settings.</div>", unsafe_allow_html=True)

# ---------------------------
# HOME
# ---------------------------
if page == "Home":
    st.markdown("<div class='top-hero'><div class='brand'>Chrono</div><div class='slogan'>BUILD YOUR CAREER BETTER</div><div class='muted'>Personal chat, job info, resume builder, mock interviews — all in one.</div></div>", unsafe_allow_html=True)
    handshake_url = "https://assets7.lottiefiles.com/packages/lf20_1pxqjqps.json"
    lhtml = "<script src='https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js'></script>"
    lhtml += "<lottie-player src='" + handshake_url + "' background='transparent' speed='1' style='width:100%;height:360px' loop autoplay></lottie-player>"
    st.components.v1.html(lhtml, height=360)

# ---------------------------
# Chrono AI (personal chat) - fast replies
# ---------------------------
if page == "Chrono AI":
    st.markdown("<div class='card'><h3 style='color:#9fe8ff'>Chrono — Personal Chat</h3><div class='muted'>Ask career questions; fast concise answers.</div></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 1.0])
    with left:
        st.markdown("<div class='chat-panel'>", unsafe_allow_html=True)
        for msg in st.session_state["chrono_msgs"]:
            role = msg.get("role")
            content = msg.get("content","")
            safe_html = html.escape(content).replace("\n","<br>")
            if role == "assistant":
                st.markdown(f"<div class='bubble-chrono'><b>Chrono</b><br>{safe_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bubble-user'><b>You</b><br>{safe_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='side-panel'><b>Recent</b></div>", unsafe_allow_html=True)
        recent = [m for m in st.session_state["chrono_msgs"] if m.get("role") == "user"]
        for r in reversed(recent[-6:]):
            st.markdown("<div style='padding:8px;border-radius:8px;background:rgba(255,255,255,0.02);margin-top:8px'>" + html.escape(r.get("content","")) + "</div>", unsafe_allow_html=True)
        with st.form("chrono_form", clear_on_submit=True):
            st.markdown("<div class='input-row'>", unsafe_allow_html=True)
            st.markdown("<div class='center-input'>", unsafe_allow_html=True)
            user_input = st.text_input("", key="chrono_input", placeholder="Ask Chrono about career paths, roles, skills, or interview tips...", label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)
            send = st.form_submit_button("➤")
            st.markdown("</div>", unsafe_allow_html=True)
        if send and user_input and user_input.strip():
            q = user_input.strip()
            st.session_state["chrono_msgs"].append({"role":"user","content":q})
            placeholder = "Composing a concise career-focused answer..."
            st.session_state["chrono_msgs"].append({"role":"assistant","content":placeholder})
            loader = st.empty()
            loader.markdown("<div class='tiny-loader'><div class='dot'></div><div class='dot2 dot'></div><div class='dot3 dot'></div></div>", unsafe_allow_html=True)
            start = time.time()
            sys_instr = "You are Chrono, an expert career guidance assistant. Answer concisely and include sample roles, required skills, recommended certifications, and example companies when relevant."
            answer = query_generative(q, system_instruction=sys_instr, model_id=st.session_state.get("model_id",""))
            elapsed = time.time() - start
            if elapsed < 0.4:
                time.sleep(0.4 - elapsed)
            # replace placeholder with answer
            for i in range(len(st.session_state["chrono_msgs"])-1, -1, -1):
                if st.session_state["chrono_msgs"][i].get("role")=="assistant" and st.session_state["chrono_msgs"][i].get("content")==placeholder:
                    st.session_state["chrono_msgs"][i]["content"] = answer
                    break
            loader.empty()

# ---------------------------
# Job Info
# ---------------------------
if page == "Job Info":
    st.markdown("<div class='card'><h3 style='color:#9fe8ff'>Job Info</h3><div class='muted'>Structured job listings and company websites.</div></div>", unsafe_allow_html=True)
    domain = st.text_input("Domain / Job title (e.g., Data Scientist)", key="job_domain")
    skills = st.text_input("Your skills (comma separated) — optional", key="job_skills")
    location = st.text_input("Preferred location — optional", key="job_location")
    max_results = st.slider("Max results", 1, 8, 3)
    if st.button("Generate Job Info"):
        if not domain.strip():
            st.error("Please enter a domain or job title.")
        else:
            st.session_state["job_history"].append({"domain":domain,"time":str(datetime.datetime.now())})
            loader_url = "https://assets3.lottiefiles.com/packages/lf20_q5pk6p1k.json"
            st.components.v1.html("<script src='https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js'></script>" +
                                 "<lottie-player src='" + loader_url + "' background='transparent' speed='1' style='width:100%;height:140px;' loop autoplay></lottie-player>", height=140)
            sys_instr = ("You are a job info assistant. OUTPUT ONLY a JSON array. Each element must be an object with keys: company_name, company_website, role_title, seniority, salary_fresher, salary_experienced, location, expected_CGPA, required_skills (array), recommended_certifications (array). No extra commentary.")
            prompt = f"Return up to {max_results} job entries for domain '{domain}' near '{location}'. Candidate skills: {skills}."
            raw = query_generative(prompt, system_instruction=sys_instr, model_id=st.session_state.get("model_id",""))
            parsed = parse_json_fallback(raw)
            if not isinstance(parsed, list):
                st.warning("Model did not return structured JSON. Showing raw output.")
                st.text_area("Raw output", value=raw, height=260)
            else:
                rows = []
                for it in parsed[:max_results]:
                    if isinstance(it, dict):
                        rows.append({
                            "Company": it.get("company_name","N/A"),
                            "Website": it.get("company_website","N/A"),
                            "Role": it.get("role_title","N/A"),
                            "Seniority": it.get("seniority","N/A"),
                            "Salary (Fresher)": it.get("salary_fresher","N/A"),
                            "Salary (Experienced)": it.get("salary_experienced","N/A"),
                            "Location": it.get("location","N/A"),
                            "Expected CGPA": it.get("expected_CGPA","N/A"),
                            "Skills": ", ".join(it.get("required_skills",[])) if isinstance(it.get("required_skills",[]), list) else str(it.get("required_skills","")),
                            "Certificates": ", ".join(it.get("recommended_certifications",[])) if isinstance(it.get("recommended_certifications",[]), list) else str(it.get("recommended_certifications",""))
                        })
                if rows:
                    if pd:
                        df = pd.DataFrame(rows)
                        st.markdown("<div class='table-area'>", unsafe_allow_html=True)
                        st.dataframe(df, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.session_state["last_job_table"] = df.to_dict(orient="records")
                    else:
                        st.write(rows)
                        st.session_state["last_job_table"] = rows

# ---------------------------
# Resume Builder
# ---------------------------
if page == "Resume Builder":
    st.markdown("<div class='card'><h3 style='color:#9fe8ff'>Resume Builder</h3><div class='muted'>Build from inputs or upload & enhance; preview then download PDF.</div></div>", unsafe_allow_html=True)
    mode = st.radio("Mode", ["Build from Input", "Upload & Enhance"], index=0)
    if mode == "Build from Input":
        with st.form("build_form"):
            full_name = st.text_input("Full name (with initials)")
            phone = st.text_input("Phone number")
            email = st.text_input("Email")
            linkedin = st.text_input("LinkedIn URL")
            degree = st.text_input("Degree (e.g., B.Tech in Computer Science)")
            institution = st.text_input("College / University")
            year = st.text_input("Graduation year")
            cgpa = st.text_input("CGPA / Percentage")
            languages = st.text_input("Languages (comma separated)")
            certifications = st.text_input("Certifications (comma separated)")
            projects = st.text_area("Projects (each line: Title - short description)")
            skills = st.text_input("Skills (comma separated)")
            address = st.text_area("Address (optional)")
            submit_build = st.form_submit_button("Preview & Generate PDF")
        if submit_build:
            details = f"Name: {full_name}\nDegree: {degree} at {institution} ({year})\nCGPA: {cgpa}\nSkills: {skills}\nProjects: {projects}\nCerts: {certifications}"
            objective = query_generative("Write a concise 2-3 sentence resume objective based on:\n\n" + details, system_instruction="Write a professional resume summary.", model_id=st.session_state.get("model_id",""))
            lang_list = [l.strip() for l in languages.split(",") if l.strip()]
            cert_list = [c.strip() for c in certifications.split(",") if c.strip()]
            skill_list = [s.strip() for s in skills.split(",") if s.strip()]
            proj_objs = []
            for line in projects.splitlines():
                if "-" in line:
                    t, d = line.split("-",1)
                    proj_objs.append({"title": t.strip(), "desc": d.strip()})
                elif line.strip():
                    proj_objs.append({"title": line.strip(), "desc": ""})
            education = [{"degree": degree, "institution": institution, "year": year, "cgpa": cgpa}]
            resume_data = {
                "full_name": full_name,
                "phone": phone,
                "email": email,
                "linkedin": linkedin,
                "address": address,
                "languages": lang_list,
                "certifications": cert_list,
                "education": education,
                "projects": proj_objs,
                "skills": skill_list,
                "objective": objective
            }
            st.markdown("<div class='card'><b>Preview</b></div>", unsafe_allow_html=True)
            st.write(f"**{full_name}** — {degree} at {institution} ({year})")
            st.write(objective)
            st.write("Skills:", ", ".join(skill_list))
            st.write("Projects:")
            for p in proj_objs:
                st.write(f"- {p.get('title')}: {p.get('desc')}")
            try:
                pdf_buf = generate_resume_pdf_bytes(resume_data)
                st.download_button("Download Resume PDF", data=pdf_buf, file_name=f"{full_name.replace(' ','_')}_resume.pdf", mime="application/pdf")
            except Exception as e:
                st.error("PDF generation not available: " + str(e))
    else:
        uploaded = st.file_uploader("Upload resume (txt/docx/pdf) to enhance", type=["txt","docx","pdf"])
        if uploaded:
            raw = uploaded.read()
            text = ""
            if uploaded.name.lower().endswith(".txt"):
                try:
                    text = raw.decode("utf-8")
                except Exception:
                    text = raw.decode("latin-1", errors="ignore")
            elif uploaded.name.lower().endswith(".docx"):
                try:
                    from docx import Document
                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
                    tmpf.write(raw); tmpf.flush()
                    doc = Document(tmpf.name)
                    text = "\n".join([p.text for p in doc.paragraphs])
                except Exception as e:
                    text = f"[docx read error: {e}]"
            else:
                text = "[PDF uploaded — paste text below for best enhancement]"
            edited = st.text_area("Editable resume text (edit only what you want changed)", value=text, height=360)
            change_instructions = st.text_area("Change instructions (what to emphasize or change)", height=120)
            if st.button("Enhance and Generate PDF"):
                sys_instr = ("You are a resume formatter. Given resume text and change instructions, OUTPUT ONLY a JSON object with keys: full_name, phone, email, linkedin, address, objective, education (array of {degree,institution,year,cgpa}), projects (array of {title,desc}), skills (array), certifications (array), languages (array). No extra commentary.")
                prompt = f"Resume text:\n{edited}\n\nChange instructions:\n{change_instructions}\n\nReturn JSON only."
                raw_out = query_generative(prompt, system_instruction=sys_instr, model_id=st.session_state.get("model_id",""))
                parsed = parse_json_fallback(raw_out)
                if isinstance(parsed, dict):
                    resume_data = {
                        "full_name": parsed.get("full_name","Candidate"),
                        "phone": parsed.get("phone",""),
                        "email": parsed.get("email",""),
                        "linkedin": parsed.get("linkedin",""),
                        "address": parsed.get("address",""),
                        "languages": parsed.get("languages",[]),
                        "certifications": parsed.get("certifications",[]),
                        "education": parsed.get("education",[]),
                        "projects": parsed.get("projects",[]),
                        "skills": parsed.get("skills",[]),
                        "objective": parsed.get("objective","")
                    }
                    try:
                        pdf_buf = generate_resume_pdf_bytes(resume_data)
                        st.download_button("Download Enhanced Resume (PDF)", data=pdf_buf, file_name="enhanced_resume.pdf", mime="application/pdf")
                        st.success("Enhanced resume generated.")
                    except Exception as e:
                        st.error("PDF generation failed: " + str(e))
                else:
                    st.warning("Could not parse model JSON. Showing raw output.")
                    st.text_area("Model raw output", value=raw_out, height=300)

# ---------------------------
# Mock Interview: Aptitude, Technical, Coding, HR Voice (live)
# ---------------------------
if page == "Mock Interview":
    st.markdown("<div class='card'><h3 style='color:#9fe8ff'>Mock Interview</h3><div class='muted'>Aptitude, Technical MCQs, Coding round, and HR voice (speak to AI).</div></div>", unsafe_allow_html=True)
    tabs = st.tabs(["Aptitude", "Technical MCQ", "Coding", "HR Voice"])
    # Aptitude: auto 25 unique questions
    with tabs[0]:
        st.info("Aptitude MCQ — 25 unique visible option buttons. Questions won't repeat until the bank is exhausted.")
        if st.button("Start Aptitude (25 questions)"):
            st.session_state["apt_quiz"] = sample_unique_apti(25)
            st.session_state["apt_answers"] = {}
        if st.session_state.get("apt_quiz"):
            for i, q in enumerate(st.session_state["apt_quiz"]):
                st.markdown(f"**Q{i+1}. {q['q']}**")
                cols = st.columns(2)
                for idx, opt in enumerate(q["options"]):
                    label = f"{chr(65+idx)}. {opt}"
                    clicked = cols[idx % 2].button(label, key=f"apt_{q['id']}_{idx}")
                    if clicked:
                        st.session_state.setdefault("apt_answers", {})[i] = idx
                sel = st.session_state.get("apt_answers", {}).get(i, None)
                if sel is not None:
                    st.markdown(f"*Selected:* **{chr(65+sel)}. {q['options'][sel]}**")
            if st.button("Submit Aptitude Answers"):
                quiz = st.session_state.get("apt_quiz", [])
                answers = st.session_state.get("apt_answers", {})
                total = len(quiz)
                correct = 0
                wrongs = []
                for i,q in enumerate(quiz):
                    u = answers.get(i)
                    if u is None or u != q["answer"]:
                        wrongs.append((i,q,u))
                    else:
                        correct += 1
                pct = round(100*correct/total) if total else 0
                if pct >= 80:
                    st.success(f"Excellent! Score: {correct}/{total} ({pct}%)")
                elif pct >= 60:
                    st.info(f"Good. Score: {correct}/{total} ({pct}%)")
                else:
                    st.warning(f"Keep practicing. Score: {correct}/{total} ({pct}%)")
                st.subheader("Review")
                for (i,q,u) in wrongs:
                    st.markdown(f"**Q{i+1}. {q['q']}**")
                    st.write(f"- Correct: **{chr(65+q['answer'])}. {q['options'][q['answer']]}**")
                    st.write(f"- Your answer: {('No answer' if u is None else chr(65+u) + '. ' + q['options'][u])}")
                    st.write(f"- Explanation: {q.get('explanation','')}")
                st.session_state["apt_quiz"] = []
                st.session_state["apt_answers"] = {}

    # Technical MCQ: auto 25 unique across chosen topics
    with tabs[1]:
        st.info("Technical MCQ — 25 unique questions across selected topics. Visible option buttons; no repeats within the session.")
        topics = list(TECH_MCQ_BANKS.keys())
        # allow multi-select topics; default all
        selected_topics = st.multiselect("Choose topics (default: all)", topics, default=topics)
        if st.button("Start Technical MCQ (25 questions)"):
            # if chosen topics have too few questions total, sampling function will reuse after clearing used ids
            st.session_state["tech_quiz"] = sample_unique_multi_topic(selected_topics, 25)
            st.session_state["tech_answers"] = {}
        if st.session_state.get("tech_quiz"):
            qz = st.session_state["tech_quiz"]
            for i,q in enumerate(qz):
                st.markdown(f"**Q{i+1}. {q['q']}**")
                cols = st.columns(2)
                for idx,opt in enumerate(q["options"]):
                    label = f"{chr(65+idx)}. {opt}"
                    clicked = cols[idx % 2].button(label, key=f"tech_{i}_{q['id']}_{idx}")
                    if clicked:
                        st.session_state.setdefault("tech_answers", {})[i] = idx
                sel = st.session_state.get("tech_answers", {}).get(i, None)
                if sel is not None:
                    st.markdown(f"*Selected:* **{chr(65+sel)}. {q['options'][sel]}**")
            if st.button("Submit Technical MCQ Answers"):
                quiz = st.session_state.get("tech_quiz", [])
                answers = st.session_state.get("tech_answers", {})
                total = len(quiz)
                correct = 0
                wrongs = []
                for i,q in enumerate(quiz):
                    u = answers.get(i)
                    if u is None or u != q["answer"]:
                        wrongs.append((i,q,u))
                    else:
                        correct += 1
                pct = round(100*correct/total) if total else 0
                if pct >= 80:
                    st.success(f"Great job! {correct}/{total} ({pct}%)")
                elif pct >= 60:
                    st.info(f"Good effort: {correct}/{total} ({pct}%)")
                else:
                    st.warning(f"Keep practicing: {correct}/{total} ({pct}%)")
                st.subheader("Review & Explanations")
                for (i,q,u) in wrongs:
                    st.markdown(f"**Q{i+1}. {q['q']}**")
                    st.write(f"- Correct: **{chr(65+q['answer'])}. {q['options'][q['answer']]}**")
                    st.write(f"- Your answer: {('No answer' if u is None else chr(65+u)+'. '+q['options'][u])}")
                    st.write(f"- Explanation: {q.get('explanation','')}")
                st.session_state["tech_quiz"] = []
                st.session_state["tech_answers"] = {}

    # Coding tab (same as before)
    with tabs[2]:
        st.header("Technical Coding Round")
        st.write("Generate a coding problem and run your Python solution against sample testcases.")
        if st.button("Get Coding Problem"):
            sys_instr = "Return a JSON object with keys: title, statement, input_format, output_format, testcases (array of {input, output}). Output only JSON."
            raw = query_generative("Generate a beginner-interview Python coding problem.", system_instruction=sys_instr, model_id=st.session_state.get("model_id",""))
            parsed = parse_json_fallback(raw)
            if isinstance(parsed, dict):
                st.session_state["coding_problem"] = parsed
            else:
                st.session_state["coding_problem"] = {
                    "title":"Sum of Even Numbers",
                    "statement":"Given N and N integers, print sum of even numbers.",
                    "input_format":"N then N integers",
                    "output_format":"single integer",
                    "testcases":[{"input":"5\n1 2 3 4 5\n","output":"6\n"}]
                }
        if st.session_state.get("coding_problem"):
            p = st.session_state["coding_problem"]
            st.subheader(p.get("title","Problem"))
            st.write(p.get("statement",""))
            st.write("*Input format:*", p.get("input_format",""))
            st.write("*Output format:*", p.get("output_format",""))
            for tc in p.get("testcases", []):
                st.code("INPUT:\n" + tc["input"] + "\nEXPECTED:\n" + tc["output"])
            code = st.text_area("Write your Python solution (reads from stdin)", value="# write code here", height=260)
            if st.button("Run & Evaluate Code"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8")
                tmp.write(code); tmp.flush(); tmp.close()
                passed = 0
                for i, tc in enumerate(p.get("testcases", [])):
                    try:
                        proc = subprocess.run([sys.executable, tmp.name], input=tc["input"].encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=6)
                        out = proc.stdout.decode("utf-8")
                        exp = tc["output"]
                        if out.strip() == exp.strip():
                            st.success(f"Testcase {i+1} passed")
                            passed += 1
                        else:
                            st.error(f"Testcase {i+1} failed. Expected {exp!r}, got {out!r}")
                            if proc.stderr:
                                st.code(proc.stderr.decode("utf-8"))
                    except subprocess.TimeoutExpired:
                        st.error(f"Testcase {i+1} timeout")
                st.write(f"Score: {passed}/{len(p.get('testcases', []))}")
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    # HR Voice tab: live mic capture (webrtc) or fallback to upload
    with tabs[3]:
        st.header("HR Voice Round — Speak to AI")
        st.write("Chrono will ask an HR question. Click 'Start Mic' (allow microphone). Press 'Record Answer' for ~5 seconds. The answer will be transcribed & evaluated.")
        if st.button("Ask HR Question (AI)"):
            q = query_generative("Provide a concise HR interview question for a software engineering candidate.", system_instruction="Be concise.", model_id=st.session_state.get("model_id",""))
            st.session_state["hr_question"] = q
            # play TTS if possible
            audio = tts_bytes_gtts(q or "Here is your question")
            if audio:
                st.audio(audio, format="audio/mp3")
            else:
                st.info("TTS not available. Install gTTS for playback.")
        st.markdown("**Question:**")
        st.write(st.session_state.get("hr_question", "[Click 'Ask HR Question (AI)']"))
        st.markdown("---")
        if webrtc_streamer is None:
            st.warning("Live microphone capture requires streamlit-webrtc. Fallback: upload recorded audio file.")
            uploaded_audio = st.file_uploader("Upload recorded answer (wav/mp3)", type=["wav","mp3"])
            if uploaded_audio and st.button("Transcribe & Evaluate Upload"):
                data = uploaded_audio.read()
                tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                try:
                    if AudioSegment is not None and uploaded_audio.type != "audio/wav":
                        seg = AudioSegment.from_file(BytesIO(data))
                        seg.export(tmp_path, format="wav")
                    else:
                        with open(tmp_path, "wb") as f:
                            f.write(data)
                    with st.spinner("Transcribing..."):
                        txt = transcribe_audio_file(tmp_path)
                    st.markdown("**Transcription:**")
                    st.write(txt or "[no transcription]")
                    eval_prompt = f"You are an HR evaluator. Evaluate the answer for clarity, communication, and content. Provide short feedback and a numeric score out of 10.\nQuestion: {st.session_state.get('hr_question','N/A')}\nAnswer: {txt}"
                    eval_res = query_generative(eval_prompt, system_instruction="Be concise and include a numeric score out of 10.", model_id=st.session_state.get("model_id",""))
                    st.subheader("Evaluation")
                    st.write(eval_res)
                    tts_eval = tts_bytes_gtts(eval_res or "No evaluation available")
                    if tts_eval:
                        st.audio(tts_eval, format="audio/mp3")
                except Exception as e:
                    st.error("Audio processing failed: " + str(e))
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
        else:
            # webrtc capture UI
            st.info("Allow mic in browser. Press 'Start Mic' then 'Record Answer' to capture your spoken reply.")
            webrtc_ctx = webrtc_streamer(key="hr_webrtc", mode=WebRtcMode.SENDONLY, audio_receiver_size=1024)
            if webrtc_ctx and webrtc_ctx.state.playing:
                if st.button("Record Answer (~5s)"):
                    st.info("Recording for 5 seconds...")
                    frames = []
                    start_t = time.time()
                    # Collect frames for 5 seconds
                    while time.time() - start_t < 5:
                        try:
                            f = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                            frames.extend(f)
                        except Exception:
                            pass
                    if not frames:
                        st.warning("No audio captured. Try again or use upload fallback.")
                    else:
                        # Convert collected frames into a WAV file
                        # frames are av.AudioFrame objects
                        try:
                            # Collect numpy arrays and metadata
                            arrays = []
                            sample_rate = None
                            nchannels = None
                            for af in frames:
                                arr = af.to_ndarray()
                                # to_ndarray returns (channels, samples)
                                if sample_rate is None:
                                    sample_rate = af.sample_rate
                                if arr.ndim == 1:
                                    ch = 1
                                else:
                                    ch = arr.shape[0]
                                nchannels = ch
                                # Convert to int16 if needed
                                if arr.dtype != np.int16:
                                    # scale floats -> int16 or cast
                                    if np.issubdtype(arr.dtype, np.floating):
                                        arr = (arr * 32767).astype(np.int16)
                                    else:
                                        arr = arr.astype(np.int16)
                                arrays.append(arr)
                            # Interleave and write to wav
                            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                            with wave.open(tmp_wav.name, "wb") as wf:
                                wf.setnchannels(nchannels or 1)
                                wf.setsampwidth(2)  # int16
                                wf.setframerate(sample_rate or 48000)
                                # write each frame's bytes sequentially
                                for a in arrays:
                                    if a.ndim == 1:
                                        wf.writeframes(a.tobytes())
                                    else:
                                        # a is (channels, samples) -> interleave
                                        interleaved = np.vstack(a).T.flatten()
                                        wf.writeframes(interleaved.tobytes())
                            # transcribe
                            with st.spinner("Transcribing..."):
                                txt = transcribe_audio_file(tmp_wav.name)
                            st.markdown("**Transcription:**")
                            st.write(txt or "[no transcription]")
                            eval_prompt = f"You are an HR evaluator. Evaluate the answer for clarity, communication, and content. Provide short feedback and a numeric score out of 10.\nQuestion: {st.session_state.get('hr_question','N/A')}\nAnswer: {txt}"
                            eval_res = query_generative(eval_prompt, system_instruction="Be concise and provide numeric score out of 10.", model_id=st.session_state.get("model_id",""))
                            st.subheader("Evaluation")
                            st.write(eval_res)
                            tts_eval = tts_bytes_gtts(eval_res or "No evaluation available")
                            if tts_eval:
                                st.audio(tts_eval, format="audio/mp3")
                        except Exception as e:
                            st.error("Failed to process recorded audio: " + str(e))
                        finally:
                            try:
                                os.unlink(tmp_wav.name)
                            except Exception:
                                pass

# ---------------------------
# Settings
# ---------------------------
if page == "Settings":
    st.header("Settings & Diagnostics")
    st.info("Paste API key for live model responses (session only). Optionally supply a model id. Upload a background image for overlay.")
    with st.form("settings_form"):
        api_input = st.text_input("API key (session only)", value=st.session_state.get("api_key",""), type="password")
        model_input = st.text_input("Optional model id (leave blank to auto-discover)", value=st.session_state.get("model_id",""))
        save = st.form_submit_button("Save")
    if save:
        st.session_state["api_key"] = api_input.strip()
        st.session_state["model_id"] = model_input.strip()
        st.success("Saved to session.")
    st.markdown("---")
    st.markdown("Upload a dark background image to overlay on the animated background (session only).")
    bgfile = st.file_uploader("Background image (png/jpg)", type=["png","jpg","jpeg"])
    if bgfile:
        data = bgfile.read()
        b64 = base64.b64encode(data).decode("utf-8")
        mime = "image/png" if bgfile.type == "image/png" else "image/jpeg"
        data_url = f"data:{mime};base64,{b64}"
        st.session_state["custom_bg_dataurl"] = data_url
        inject_css(data_url)
        st.success("Background applied for this session.")
    st.markdown("---")
    if st.button("Reset MCQ history (allow repeats)"):
        st.session_state["used_apt_ids"] = set()
        st.session_state["used_tech_ids"] = set()
        st.success("MCQ history cleared.")
    st.markdown("Diagnostics")
    st.write("API key set:", bool(st.session_state.get("api_key","")))
    st.write("Model id (session):", st.session_state.get("model_id",""))
    st.write("Used aptitude ids count:", len(st.session_state.get("used_apt_ids", set())))
    st.write("Used technical ids count:", len(st.session_state.get("used_tech_ids", set())))
    st.write("Generative client available:", bool(genai))
    st.write("WeRTC available:", bool(webrtc_streamer))
    st.write("Reportlab available:", bool(SimpleDocTemplate))
    st.write("gTTS available:", bool(gTTS))
    st.write("whisper available:", bool(whisper))
    st.write("SpeechRecognition available:", bool(sr))

# End of file
