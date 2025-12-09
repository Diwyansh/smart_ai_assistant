import json
import time
from docx import Document
from typing import List, Dict, Any, Set
import pdfplumber
import pyttsx3
import streamlit as st
import pandas as pd
from openai import OpenAI
from openai import APIStatusError
import sounddevice as sd
import numpy as np
import speech_recognition as sr

# Try import mic recorder (install via pip install streamlit-mic-recorder)
try:
    from streamlit_mic_recorder import mic_recorder, speech_to_text
    MIC_AVAILABLE = True
except Exception:
    MIC_AVAILABLE = False
    def speech_to_text(**kwargs):
        return None

# -------------------------------------------------------------
# Page config + header (use default streamlit theme/background)
# -------------------------------------------------------------
st.set_page_config(page_title="Clinical Document Compare + CRA Chat", layout="wide")

# Small CSS to make the chat input + mic look cohesive (doesn't change page background)
st.markdown(
    """
    <style>
    /* GLOBAL WRAPPER – keep 70% layout */
    .main-container {
        max-width: 70% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .block-container {
        max-width: 70% !important;
        margin-left: auto !important;
        margin-right: auto !important;
           padding-top: 1rem !important;
    padding-bottom: 0rem !important;
    }

    /* Chat container scroll */
    .chat-wrapper {
        width: 100%;
        margin: auto;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding-bottom: 60px;
    }

    /* Assistant/user bubble styling */
    .chat-bubble-assistant {
        width : 100%;
        border-radius: 10px;
        padding: 10px;
        margin: auto;
        margin-bottom: 8px;
    }
    .chat-bubble-user {
        width : 100%;
        border-radius: 10px;
        padding: 10px;
        margin-left: auto;
        text-align: right;
        margin-bottom: 8px;
    }

    /* Sticky input row */
    .chat-input-row {
        position: sticky;
        bottom: 0;
        display: flex;
        gap: 8px;
        align-items: center;
        padding: 8px 0;
    }

    .chat-input {
        width: 100%;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid #D1D5DB;
        font-size: 16px;
    }

    .mic-button {
        background: none;
        border: none;
        cursor: pointer;
        width: 40px;
        height: 40px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    div[data-testid="stTextInput"] {
    margin-bottom: 20px;
}
    .stColumn:empty {
    display: none !important;
}
    </style>
    """, unsafe_allow_html=True
)

# -------------------------------------------------------------
# OpenAI Client
# -------------------------------------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# -------------------------------------------------------------
# ORIGINAL FUNCTIONS (unchanged)
# -------------------------------------------------------------
def read_doc_structured(uploaded_file) -> List[Dict[str, str]]:
    document = Document(uploaded_file)
    structured_content: List[Dict[str, Any]] = []
    current_section = "Document"

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if 'Heading' in para.style.name:
            current_section = text
            structured_content.append({"section_title": current_section, "text": []})
        else:
            if structured_content:
                structured_content[-1]['text'].append(text)
            else:
                structured_content.append({"section_title": "Document", "text": []})
                structured_content[-1]['text'].append(text)

    final_sections: List[Dict[str, str]] = []
    for section in structured_content:
        if section['text']:
            final_sections.append({
                "section_title": section['section_title'],
                "text": '\n'.join(section['text'])
            })
    return final_sections

def read_pdf_structured(uploaded_file):
    """Extract text from a PDF and return as a single section."""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return [{"section_title": "PDF_CONTENT", "text": f"ERROR reading PDF: {e}"}]

    return [{"section_title": "PDF_CONTENT", "text": text}]

def read_file_structured(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".docx"):
        return read_doc_structured(uploaded_file)
    elif filename.endswith(".pdf"):
        return read_pdf_structured(uploaded_file)

    return [{"section_title": "UNKNOWN_FORMAT", "text": ""}]

def analyze_sections(old_text: str, new_text: str, section_title: str) -> Dict[str, Any]:

    schema = """
[
  {
    "section_title": "string",
    "change_summary": "string",
    "category": "string",
    "clinical_impact": "string",
    "severity": "string"
  }
]
"""

    system_prompt = (
        f"You are an expert Clinical Research Associate. Your task is to compare two text segments "
        f"from a clinical trial protocol. Identify *only* clinically significant changes. "
        f"Ignore formatting, spelling, or minor stylistic differences.\n\n"
        f"You MUST return a JSON array that strictly conforms to:\n{schema}\n"
        f"Return ONLY raw JSON. No explanation."
    )

    user_prompt = (
        f"Analyze changes in the section titled: '{section_title}'.\n\n"
        f"old text\n{old_text}\n\n"
        f"new text\n{new_text}\n\n"
        "Identify clinically meaningful differences."
    )

    snippet_len = 100
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="tngtech/deepseek-r1t2-chimera:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
            )

            json_string = response.choices[0].message.content

            if json_string.startswith("```json"):
                json_string = json_string.strip()
                json_string = json_string[len("```json"):].strip()
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].strip()

            report_data = json.loads(json_string)

            validated_changes: List[Dict[str, Any]] = []

            if isinstance(report_data, dict) and 'changes' in report_data:
                change_list = report_data['changes']
            elif isinstance(report_data, list):
                change_list = report_data
            else:
                change_list = []

            for change in change_list:
                required_keys = ['category', 'change_summary', 'severity', 'clinical_impact']
                if all(key in change for key in required_keys):
                    validated_changes.append(change)

            return {
                "title": section_title,
                "old_text": old_text,
                "new_text": new_text,
                "changes": validated_changes,
                "status": "ANALYSIS_COMPLETE",
                "old_snippet": old_text[:snippet_len] + "...",
                "new_snippet": new_text[:snippet_len] + "...",
            }

        except APIStatusError as e:
            if e.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                raise

        except Exception as e:
            raise e

    return {
        "title": section_title,
        "old_text": old_text,
        "new_text": new_text,
        "changes": [],
        "status": "ANALYSIS_FAILED",
        "old_snippet": old_text[:snippet_len] + "...",
        "new_snippet": new_text[:snippet_len] + "...",
    }

def submit_query():
    user_query_text = st.session_state.user_query_input
    if user_query_text:
        st.session_state.chat_history.append({"role": "user", "content": user_query_text})
        placeholder_index = len(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": "⏳ Thinking..."})
        st.session_state.pending_llm = {
            "placeholder_index": placeholder_index,
            "query_text": user_query_text
        }
        st.session_state.user_query_input = ""  # clear box
        st.rerun()

def record_and_transcribe(threshold=0.01, silence_limit=2, max_duration=20, fs=16000, chunk=1024):
    """
    Dynamically records audio until user stops speaking OR max_duration is reached.
    - threshold: minimum audio amplitude to detect voice
    - silence_limit: seconds of continuous silence to stop recording
    - max_duration: maximum total recording duration in seconds
    """
    r = sr.Recognizer()
    audio_buffer = []

    placeholder = st.empty()
    time.sleep(0.2)
    placeholder.markdown("<div style='text-align:center; font-size:20px;'>🔴 Listening...</div>", unsafe_allow_html=True)

    def callback(indata, frames, time_info, status):
        audio_buffer.append(indata.copy())

    try:
        stream = sd.InputStream(samplerate=fs, channels=1, callback=callback, blocksize=chunk)
        stream.start()

        silence_start_time = None
        start_time = time.time()  # track total recording time

        while True:
            if len(audio_buffer) == 0:
                time.sleep(0.05)
                continue

            # Stop if max duration reached
            if time.time() - start_time >= max_duration:
                break

            last_chunk = audio_buffer[-1].flatten()
            amplitude = np.max(np.abs(last_chunk))

            if amplitude > threshold:
                silence_start_time = None
            else:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time >= silence_limit:
                    break

            silent_seconds = 0 if silence_start_time is None else time.time() - silence_start_time
            elapsed_seconds = time.time() - start_time
            placeholder.markdown(
                f"<div style='text-align:center; font-size:20px;'>🔴</div>",
                unsafe_allow_html=True
            )

            time.sleep(0.05)

        stream.stop()
        stream.close()
        placeholder.markdown("<div style='text-align:center; font-size:20px;'>⏳</div>", unsafe_allow_html=True)

        audio_data = np.concatenate(audio_buffer, axis=0)
        audio_data = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_data.tobytes()
        audio = sr.AudioData(audio_bytes, fs, 2)

        try:
            text = r.recognize_google(audio)
            placeholder.empty()
            return text
        except sr.UnknownValueError:
            placeholder.empty()
            st.warning("Could not understand audio")
            return ""
        except sr.RequestError:
            placeholder.empty()
            st.error("Speech recognition service failed")
            return ""
    except Exception as e:
        placeholder.empty()
        st.error(f"Recording failed: {e}")
        return ""

# -------------------------------------------------------------
# CRA persona (strict)
# -------------------------------------------------------------
CRA_SYSTEM_PROMPT = """
You are an expert Clinical Research Associate (CRA).
You must STRICTLY stay within the domain of:
- clinical trials
- safety, AE/SAE reporting
- eligibility criteria
- dosing
- endpoints
- study design
- interpretation of the protocol comparison results

You MUST REFUSE any question outside clinical research with:
"I'm only able to answer questions related to clinical research or your protocol comparison."

Do not answer personal, political, legal, math, or technical programming questions.
Be precise, concise, factual, and professional.
If the user asks about the comparison report, use the dataframe provided as context.
"""

# -------------------------------------------------------------
# Session state (including voice flags)
# -------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []
if "old_uploaded" not in st.session_state:
    st.session_state.old_uploaded = False
if "new_uploaded" not in st.session_state:
    st.session_state.new_uploaded = False
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if "final_report_df" not in st.session_state:
    st.session_state.final_report_df = None
if "old_file_obj" not in st.session_state:
    st.session_state.old_file_obj = None
if "new_file_obj" not in st.session_state:
    st.session_state.new_file_obj = None
if "_mic_transcribed" not in st.session_state:
    st.session_state._mic_transcribed = None
    

# Voice related state
if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False
if "voice_auto_listen" not in st.session_state:
    st.session_state.voice_auto_listen = False
if "voice_last_read_idx" not in st.session_state:
    st.session_state.voice_last_read_idx = -1
if "voice_listening_in_progress" not in st.session_state:
    st.session_state.voice_listening_in_progress = False

# -------------------------------------------------------------
# Layout and guided chat-like flow
# -------------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🧬 Clinical Document Compare — CRA Assistant</h1>", unsafe_allow_html=True)

if len(st.session_state.chat_history) == 0:
    welcome_text = (
        "**Hello — I'm your Clinical Research Associate Assistant.**\n\n"
        "I'll guide you through uploading two protocol versions and will run a clinical comparison.\n\n"
        "**Step 1:** Please upload **Version 1** using the upload control below."
    )
    st.session_state.chat_history.append({"role": "assistant", "content": welcome_text})

with st.container():
    chat_html = "<div class='chat-container'>"
    for idx, msg in enumerate(st.session_state.chat_history):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")

        if role == "assistant":
            st.markdown(content, unsafe_allow_html=True)

            if st.button("🔊 Listen", key=f"tts_{idx}"):
                st.session_state["tts_index"] = idx

        else:
            st.markdown(f"<div style='text-align:right'>{content}</div>", unsafe_allow_html=True)
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎙️ Voice Conversation Controls")
    col_v1, col_v2 = st.columns([0.6, 0.4])
    with col_v1:
        voice_mode_checkbox = st.checkbox("Enable Voice Mode (read assistant messages & listen to you)", value=st.session_state.voice_mode, key="voice_mode")
    with col_v2:
        auto_listen_checkbox = st.checkbox("Auto-listen after assistant speech", value=st.session_state.voice_auto_listen, key="voice_auto_listen")

    st.markdown("---")

    if not st.session_state.old_uploaded:
        old_file = st.file_uploader("📄 Upload Version 1 (OLD)", type=["docx", "pdf"], key="old_doc")
        if old_file is not None:
            st.session_state.old_uploaded = True
            st.session_state.old_file_obj = old_file
            st.session_state.chat_history.append({"role": "assistant", "content": "✅ **Version 1 (OLD) uploaded.** Now please upload **Version 2 (NEW)**."})
            st.rerun()

    elif not st.session_state.new_uploaded:
        new_file = st.file_uploader("📄 Upload Version 2 (NEW)", type=["docx", "pdf"], key="new_doc")
        if new_file is not None:
            st.session_state.new_uploaded = True
            st.session_state.new_file_obj = new_file
            st.session_state.chat_history.append({"role": "assistant", "content": "✅ **Version 2 (NEW) uploaded.** Click **Analyze Contextual Changes** when ready."})
            st.rerun()

    elif not st.session_state.analysis_done and not st.session_state.run_analysis:
        run_btn = st.button("🚀 Analyze Contextual Changes", type="primary")
        if run_btn:
            st.session_state.run_analysis = True
            st.session_state.chat_history.append({"role": "assistant", "content": "🔎 Starting analysis — this may take a short while. I'll show progress as I process each section."})
            st.rerun()

    if st.session_state.run_analysis and not st.session_state.analysis_done:
        with st.spinner("Extracting text and running semantic analysis..."):
            try:
                old_sections = read_file_structured(st.session_state.old_file_obj)
                new_sections = read_file_structured(st.session_state.new_file_obj)

                old_map: Dict[str, str] = {sec['section_title']: sec['text'] for sec in old_sections}
                new_map: Dict[str, str] = {sec['section_title']: sec['text'] for sec in new_sections}

                all_titles: Set[str] = set(old_map.keys()) | set(new_map.keys())

                st.session_state.chat_history.append({"role": "assistant", "content": f"🧩 Segmentation complete. Found {len(all_titles)} unique sections. Starting per-section semantic comparison."})

            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ File reading error: {e}"})
                st.session_state.run_analysis = False
                st.rerun()

            all_changes_to_analyze: List[Dict[str, Any]] = []
            progress_placeholder = st.empty()
            for idx, title in enumerate(sorted(list(all_titles))):
                progress = (idx + 1) / len(all_titles)
                progress_placeholder.progress(progress, text=f"Analyzing {title} ({int(progress*100)}%)")
                text_a = old_map.get(title, "SECTION DELETED (TEXT N/A)")
                text_b = new_map.get(title, "SECTION ADDED (TEXT N/A)")

                if text_a != text_b:
                    try:
                        analysis_result = analyze_sections(text_a, text_b, title)
                        all_changes_to_analyze.append(analysis_result)
                    except Exception as e:
                        all_changes_to_analyze.append({
                            "title": title,
                            "old_snippet": text_a[:100] + "...",
                            "new_snippet": text_b[:100] + "...",
                            "changes": [{"category": "SYSTEM ERROR", "change_summary": str(e), "severity": "High", "clinical_impact": "Processing failed"}],
                            "status": "ANALYSIS_FAILED",
                        })
                        st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Error analyzing section '{title}': {e}"})

            progress_placeholder.empty()

            report_data = []
            for section_analysis in all_changes_to_analyze:
                if section_analysis['status'] == "ANALYSIS_FAILED":
                    report_data.append({
                        "Section Title": section_analysis['title'],
                        "Category": "SYSTEM ERROR",
                        "Severity": "Critical",
                        "Change Summary": section_analysis['changes'][0]['change_summary'],
                        "Clinical Impact": "Analysis failed.",
                        "Old Text Snippet": section_analysis.get('old_snippet', 'N/A'),
                        "New Text Snippet": section_analysis.get('new_snippet', 'N/A'),
                    })
                elif section_analysis['changes']:
                    for detail in section_analysis['changes']:
                        report_data.append({
                            "Section Title": section_analysis['title'],
                            "Category": detail.get('category'),
                            "Severity": detail.get('severity'),
                            "Change Summary": detail.get('change_summary'),
                            "Clinical Impact": detail.get('clinical_impact'),
                            "Old Text Snippet": section_analysis['old_snippet'],
                            "New Text Snippet": section_analysis['new_snippet'],
                        })
                else:
                    report_data.append({
                        "Section Title": section_analysis['title'],
                        "Category": "Administrative / No Clinical Change",
                        "Severity": "Low",
                        "Change Summary": "Non-clinical difference only.",
                        "Clinical Impact": "No action required.",
                        "Old Text Snippet": section_analysis['old_snippet'],
                        "New Text Snippet": section_analysis['new_snippet'],
                    })

            df = pd.DataFrame(report_data)
            st.session_state.final_report_df = df

            if df is not None:
                df_html = df.to_html(index=False)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"### ✅ Context Compare Final Report (Available for Chat Context)\n{df_html}"
                })

            st.session_state.analysis_done = True
            st.session_state.run_analysis = False
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "You may now ask clinical research questions about the comparison. Use the input box below or the microphone button to speak."
            })
            st.rerun()

    if st.session_state.analysis_done:
        st.markdown("### 💬 Ask the CRA Assistant (clinical questions only)")
if "user_query_input_temp" in st.session_state:
    st.session_state["user_query_input"] = st.session_state.pop("user_query_input_temp")

if "tts_index" in st.session_state:
    idx = st.session_state.pop("tts_index")
    try:
        text_to_speak = st.session_state.chat_history[idx]["content"]
        engine = pyttsx3.init()
        engine.say(text_to_speak)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"TTS failed: {e}")

col1, col2 = st.columns([0.95, 0.05])

with col1:
    user_query_text = st.text_input(
        "",
        key="user_query_input",
        placeholder="Ask a clinical research question...",
        label_visibility="collapsed",
        value=st.session_state.get("user_query_input", ""),
        on_change=submit_query,
    )

with col2:
    if MIC_AVAILABLE and st.button("🎤", key="mic_icon_button"):
        transcribed = record_and_transcribe()
        if transcribed:
            st.session_state["user_query_input_temp"] = transcribed
            st.rerun()

if "pending_llm" in st.session_state:
    pending = st.session_state.pop("pending_llm")
    placeholder_index = pending["placeholder_index"]
    query_text = pending["query_text"]

    report_context_text = ""
    if st.session_state.final_report_df is not None:
        try:
            report_context_text = st.session_state.final_report_df.to_json()
        except Exception:
            report_context_text = ""

    messages = [
        {"role": "system", "content": CRA_SYSTEM_PROMPT},
        {"role": "system", "content": "Protocol comparison report (JSON): " + report_context_text}
    ]
    for h in st.session_state.chat_history[-20:]:
        messages.append({"role": h["role"], "content": h["content"]})

    try:
        llm_response = client.chat.completions.create(
            model="tngtech/deepseek-r1t2-chimera:free",
            temperature=0.0,
            messages=messages
        )
        response_text = llm_response.choices[0].message.content
    except Exception as e:
        response_text = f"Error contacting AI: {e}"

    st.session_state.chat_history[placeholder_index]["content"] = response_text
    st.rerun()

# ---------------------------
# Voice-mode automation logic (fixed)
# ---------------------------
def process_voice_mode():
    """
    Voice-mode: sequential TTS → auto-listen → LLM → TTS response
    Handles multiple assistant messages in a row.
    """
    if not st.session_state.voice_mode:
        return

    # Find last assistant message
    last_assist_idx = -1
    for i, msg in enumerate(st.session_state.chat_history):
        if msg.get("role") == "assistant":
            last_assist_idx = i

    if last_assist_idx <= st.session_state.get("voice_last_read_idx", -1):
        return  # nothing new

    # Initialize per-message flags
    tts_flag = f"voice_tts_done_{last_assist_idx}"
    listen_flag = f"voice_listened_done_{last_assist_idx}"
    if tts_flag not in st.session_state:
        st.session_state[tts_flag] = False
    if listen_flag not in st.session_state:
        st.session_state[listen_flag] = False

    # Step 1: TTS
    if not st.session_state[tts_flag]:
        st.session_state.voice_listening_in_progress = True
        try:
            text_to_speak = st.session_state.chat_history[last_assist_idx]["content"]
            engine = pyttsx3.init()
            engine.say(text_to_speak)
            engine.runAndWait()
            time.sleep(0.2)
            st.session_state[tts_flag] = True
        finally:
            st.session_state.voice_listening_in_progress = False
            st.rerun()
        return

    # Step 2: Auto-listen (optional)
    if st.session_state.voice_auto_listen and not st.session_state[listen_flag]:
        st.session_state.voice_listening_in_progress = True
        try:
            user_text = record_and_transcribe(max_duration=120)
            if user_text:
                st.session_state.chat_history.append({"role": "user", "content": user_text})
                placeholder_index = len(st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "assistant", "content": "⏳ Thinking..."})
                engine = pyttsx3.init()
                engine.say(
                    "I am analyzing your question and its clinically significant details. Please allow me a moment."
                )
                engine.runAndWait()
                st.session_state.pending_llm = {
                    "placeholder_index": placeholder_index,
                    "query_text": user_text
                }
            st.session_state[listen_flag] = True
        finally:
            st.session_state.voice_listening_in_progress = False
            # mark this message as fully processed
            st.session_state.voice_last_read_idx = last_assist_idx
            st.rerun()


process_voice_mode()
