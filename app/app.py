"""
app.py — SehaTrack Pro
Unified Streamlit application combining:
  • Doctor auth + patient login portal
  • Patient registration workflow
  • AI-powered NLP symptom triage (voice + text)
  • CheXNet chest X-ray analysis with Grad-CAM++ heatmaps
  • Kvasir GI endoscopy analysis with LIME explainability
  • Secure tele-chat between doctor and patient
  • Prescriptions management (doctor adds, patient views)
  • Appointment booking (patient requests, doctor confirms/rejects)
  • Follow-up reminders for patients
  • Clinical insights dashboard
  • Clinical logs with CSV export
  • Patient search
"""
import hashlib
import io



import json
import os
import sqlite3
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st  # line 33
import torch
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
import torchvision.transforms as T
import whisper
from PIL import Image

from download_models import download_all_weights
download_all_weights()   



from model import (
    DEVICE,
    DISEASE_LABELS,
    GI_CLASSES,
    OPTIMAL_THRESHOLDS,
    GradCAMPlusPlus,
    encode_meta,
    load_kvasir_engine,
    load_vision_engine,
    predict,
    predict_topk,
    run_kvasir_lime_explanation,
)

# ── Optional TF ───────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    tf = None
    _TF_AVAILABLE = False



# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="SehaTrack Pro", layout="wide", page_icon="🏥")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #f8fafc; }
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
div.stButton > button {
    background: #2563eb; color: white;
    border-radius: 10px; border: none;
    padding: 0.5rem 1.5rem; font-weight: 500;
}
div.stButton > button:hover { background: #1d4ed8; }
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input { border-radius: 8px; }
.doctor-badge {
    background: #1e3a5f; color: #93c5fd;
    padding: 4px 12px; border-radius: 20px;
    font-size: 13px; display: inline-block; margin-bottom: 8px;
}
.patient-badge {
    background: #1a3a2a; color: #86efac;
    padding: 4px 12px; border-radius: 20px;
    font-size: 13px; display: inline-block; margin-bottom: 8px;
}
.section-card {
    background: white; border-radius: 14px;
    padding: 1.5rem; border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}
.diagnosis-pill {
    background: #EFF6FF; border-left: 4px solid #3B82F6;
    padding: 12px 20px; border-radius: 8px; margin-bottom: 10px;
}
.accuracy-pill {
    background: #F0FDF4; border-left: 4px solid #22C55E;
    padding: 12px 20px; border-radius: 8px; margin-bottom: 10px;
}
.rx-card {
    background: #f0fdf4; border-left: 4px solid #22c55e;
    padding: 12px 16px; border-radius: 8px; margin-bottom: 8px;
}
.appt-pending  { background: #fefce8; border-left: 4px solid #eab308; padding: 10px 16px; border-radius: 8px; margin-bottom: 6px; }
.appt-approved{ background: #f0fdf4; border-left: 4px solid #22c55e; padding: 10px 16px; border-radius: 8px; margin-bottom: 6px; }
.appt-rejected { background: #fef2f2; border-left: 4px solid #ef4444; padding: 10px 16px; border-radius: 8px; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
_HERE   = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "sehatrack.db")


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _fetch_all(sql: str, params: tuple = ()) -> List[Dict]:
    con = _conn()
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def _fetch_one(sql: str, params: tuple = ()) -> Optional[Dict]:
    con = _conn()
    try:
        r = con.execute(sql, params).fetchone()
        return dict(r) if r else None
    finally:
        con.close()


def _write(sql: str, params: tuple = ()) -> int:
    con = _conn()
    try:
        cur = con.execute(sql, params)
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def init_db() -> None:
    con = _conn()
    cur = con.cursor()
    cur.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS doctors (
        doctor_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name     TEXT,
        specialty     TEXT,
        created_at    TEXT
    );

    CREATE TABLE IF NOT EXISTS patients (
        patient_id        TEXT PRIMARY KEY,
        full_name         TEXT,
        date_of_birth     TEXT,
        age               INTEGER,
        gender            TEXT,
        blood_type        TEXT,
        phone             TEXT,
        email             TEXT,
        insurance_status  TEXT,
        emergency_contact TEXT,
        allergies         TEXT,
        created_at        TEXT,
        username          TEXT UNIQUE,
        password_hash     TEXT
    );

    CREATE TABLE IF NOT EXISTS visits (
        visit_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id          TEXT REFERENCES patients(patient_id),
        doctor_id           INTEGER REFERENCES doctors(doctor_id),
        visit_datetime      TEXT,
        visit_type          TEXT,
        department          TEXT,
        chief_complaint     TEXT,
        asr_transcription   TEXT,
        symptom_top1        TEXT,
        symptom_top1_score  REAL,
        symptom_top2        TEXT,
        symptom_top2_score  REAL,
        symptom_top3        TEXT,
        symptom_top3_score  REAL,
        urgency             TEXT,
        severity_score      INTEGER,
        follow_up_needed    INTEGER,
        doctor_notes        TEXT,
        recommended_steps   TEXT,
        audio_duration_sec  REAL,
        processed_at        TEXT
    );

    CREATE TABLE IF NOT EXISTS diagnoses (
        diagnosis_id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id     INTEGER REFERENCES visits(visit_id),
        rank         INTEGER,
        disease      TEXT,
        likelihood   TEXT,
        reasoning    TEXT
    );

    CREATE TABLE IF NOT EXISTS medical_records (
        record_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id    TEXT REFERENCES patients(patient_id),
        doctor_id     INTEGER REFERENCES doctors(doctor_id),
        image_path    TEXT,
        view_position TEXT,
        all_findings  TEXT,
        doctor_notes  TEXT,
        timestamp     TEXT
    );

    CREATE TABLE IF NOT EXISTS chat_messages (
        msg_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        sender_id    INTEGER,
        receiver_id  INTEGER,
        message_body TEXT,
        timestamp    TEXT DEFAULT (datetime('now','localtime'))
    );

    CREATE TABLE IF NOT EXISTS prescriptions (
        prescription_id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id        INTEGER REFERENCES visits(visit_id),
        patient_id      TEXT REFERENCES patients(patient_id),
        doctor_id       INTEGER REFERENCES doctors(doctor_id),
        medication_name TEXT NOT NULL,
        dosage          TEXT,
        frequency       TEXT,
        duration        TEXT,
        notes           TEXT,
        issued_at       TEXT
    );

    CREATE TABLE IF NOT EXISTS appointments (
        appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id     TEXT REFERENCES patients(patient_id),
        doctor_id      INTEGER REFERENCES doctors(doctor_id),
        requested_date TEXT,
        requested_time TEXT,
        reason         TEXT,
        status         TEXT DEFAULT 'pending',
        doctor_notes   TEXT,
        created_at     TEXT
    );
    """)

    # Seed default admin doctor
    if cur.execute("SELECT COUNT(*) FROM doctors").fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO doctors (username,password_hash,full_name,specialty,created_at) VALUES (?,?,?,?,?)",
            ("admin", hashlib.sha256("admin123".encode()).hexdigest(),
             "Dr. Admin", "General Practice", datetime.now().isoformat()),
        )

    con.commit()

    # ── Schema migrations ─────────────────────────────────────────────────────
    def _add_col(table, col, typ):
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({table})")}
        if col not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    for col, typ in [
        ("date_of_birth", "TEXT"), ("phone", "TEXT"), ("email", "TEXT"),
        ("emergency_contact", "TEXT"), ("allergies", "TEXT"), ("created_at", "TEXT"),
        ("username", "TEXT"), ("password_hash", "TEXT"),
    ]:
        _add_col("patients", col, typ)

    for col, typ in [
        ("doctor_id", "INTEGER"), ("doctor_notes", "TEXT"), ("audio_duration_sec", "REAL"),
    ]:
        _add_col("visits", col, typ)

    # ── appointments migrations (fixes "no such column: requested_date") ──────
    for col, typ in [
        ("requested_date", "TEXT"),
        ("requested_time", "TEXT"),
        ("reason",         "TEXT"),
        ("status",         "TEXT"),
        ("doctor_notes",   "TEXT"),
        ("created_at",     "TEXT"),
    ]:
        _add_col("appointments", col, typ)

    # ── medical_records migrations ────────────────────────────────────────────
    for col, typ in [
        ("doctor_id",    "INTEGER"),
        ("image_path",   "TEXT"),
        ("view_position","TEXT"),
        ("all_findings", "TEXT"),
        ("doctor_notes", "TEXT"),
        ("timestamp",    "TEXT"),
    ]:
        _add_col("medical_records", col, typ)

    # ── chat_messages migrations ──────────────────────────────────────────────
    for col, typ in [
        ("sender_id",    "INTEGER"),
        ("receiver_id",  "INTEGER"),
        ("message_body", "TEXT"),
        ("timestamp",    "TEXT"),
    ]:
        _add_col("chat_messages", col, typ)

    # ── prescriptions migrations ──────────────────────────────────────────────
    for col, typ in [
        ("visit_id",        "INTEGER"),
        ("patient_id",      "TEXT"),
        ("doctor_id",       "INTEGER"),
        ("medication_name", "TEXT"),
        ("dosage",          "TEXT"),
        ("frequency",       "TEXT"),
        ("duration",        "TEXT"),
        ("notes",           "TEXT"),
        ("issued_at",       "TEXT"),
    ]:
        _add_col("prescriptions", col, typ)

    con.commit()
    con.close()


init_db()

# ══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def verify_login(username: str, password: str) -> Optional[Tuple]:
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT doctor_id, full_name, specialty FROM doctors WHERE username=? AND password_hash=?",
        (username, _hash(password)),
    ).fetchone()
    con.close()
    return row


def register_doctor(username: str, password: str, full_name: str, specialty: str) -> bool:
    try:
        _write(
            "INSERT INTO doctors (username,password_hash,full_name,specialty,created_at) VALUES (?,?,?,?,?)",
            (username, _hash(password), full_name, specialty, datetime.now().isoformat()),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def verify_patient_login(username: str, password: str) -> Optional[Dict]:
    return _fetch_one(
        "SELECT * FROM patients WHERE username=? AND password_hash=?",
        (username, _hash(password)),
    )


def register_patient_account(patient_id: str, username: str, password: str) -> bool:
    con = _conn()
    try:
        existing = con.execute(
            "SELECT patient_id FROM patients WHERE username=?", (username,)
        ).fetchone()
        if existing:
            return False
        con.execute(
            "UPDATE patients SET username=?, password_hash=? WHERE patient_id=?",
            (username, _hash(password), patient_id),
        )
        con.commit()
        return True
    finally:
        con.close()


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT & VISIT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _patient_id(name: str, dob: str) -> str:
    return "PAT-" + hashlib.md5(f"{name.lower().strip()}{dob}".encode()).hexdigest()[:8].upper()


def _patient_chat_id(patient_id: str) -> int:
    return int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % (10 ** 9)


def get_or_create_patient(data: Dict) -> Tuple[str, bool]:
    pid      = _patient_id(data["full_name"], data["date_of_birth"])
    existing = _fetch_one("SELECT patient_id FROM patients WHERE patient_id=?", (pid,))
    if not existing:
        _write(
            """INSERT INTO patients
               (patient_id,full_name,date_of_birth,age,gender,blood_type,
                phone,email,insurance_status,emergency_contact,allergies,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pid, data["full_name"], data["date_of_birth"], data["age"],
             data["gender"], data["blood_type"], data["phone"], data["email"],
             data["insurance_status"], data["emergency_contact"], data["allergies"],
             datetime.now().isoformat()),
        )
        return pid, False
    return pid, True


def save_visit(patient_id: str, doctor_id: int, vd: Dict, diagnoses: List) -> int:
    visit_id = _write(
        """INSERT INTO visits
           (patient_id,doctor_id,visit_datetime,visit_type,department,
            chief_complaint,asr_transcription,symptom_top1,symptom_top1_score,
            symptom_top2,symptom_top2_score,symptom_top3,symptom_top3_score,
            urgency,severity_score,follow_up_needed,doctor_notes,
            recommended_steps,audio_duration_sec,processed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (patient_id, doctor_id,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         vd.get("visit_type"), vd.get("department"),
         vd.get("chief_complaint"), vd.get("transcription"),
         vd.get("symptom_top1"), vd.get("symptom_top1_score"),
         vd.get("symptom_top2"), vd.get("symptom_top2_score"),
         vd.get("symptom_top3"), vd.get("symptom_top3_score"),
         vd.get("urgency"), vd.get("severity_score"),
         int(vd.get("follow_up_needed", False)),
         vd.get("doctor_notes"), vd.get("recommended_steps"),
         vd.get("audio_duration_sec"), datetime.now().isoformat()),
    )
    for rank, diag in enumerate(diagnoses, 1):
        _write(
            "INSERT INTO diagnoses (visit_id,rank,disease,likelihood,reasoning) VALUES (?,?,?,?,?)",
            (visit_id, rank, diag.get("disease"), diag.get("likelihood"), diag.get("reasoning")),
        )
    return visit_id


def get_patient_history(patient_id: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query("""
        SELECT v.*, d.full_name AS doctor_name, d.specialty AS doctor_specialty
        FROM visits v
        LEFT JOIN doctors d ON v.doctor_id = d.doctor_id
        WHERE v.patient_id = ?
        ORDER BY v.visit_datetime DESC
    """, con, params=(patient_id,))
    con.close()
    return df


def get_all_visits() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query("""
        SELECT v.*, p.full_name, p.age, p.gender, p.blood_type, p.insurance_status,
               d.full_name AS doctor_name
        FROM visits v
        JOIN patients p ON v.patient_id = p.patient_id
        LEFT JOIN doctors d ON v.doctor_id = d.doctor_id
    """, con)
    con.close()
    return df


# ── Prescription helpers ───────────────────────────────────────────────────────
def add_prescription(visit_id: int, patient_id: str, doctor_id: int, meds: List[Dict]) -> None:
    for med in meds:
        _write(
            """INSERT INTO prescriptions
               (visit_id,patient_id,doctor_id,medication_name,dosage,
                frequency,duration,notes,issued_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (visit_id, patient_id, doctor_id,
             med.get("medication_name"), med.get("dosage"),
             med.get("frequency"), med.get("duration"),
             med.get("notes"), datetime.now().isoformat()),
        )


def get_patient_prescriptions(patient_id: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query("""
        SELECT p.*, d.full_name AS doctor_name, v.visit_datetime
        FROM prescriptions p
        LEFT JOIN doctors d ON p.doctor_id = d.doctor_id
        LEFT JOIN visits  v ON p.visit_id  = v.visit_id
        WHERE p.patient_id = ?
        ORDER BY p.issued_at DESC
    """, con, params=(patient_id,))
    con.close()
    return df


def get_visit_prescriptions(visit_id: int) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query(
        "SELECT * FROM prescriptions WHERE visit_id=? ORDER BY issued_at DESC",
        con, params=(visit_id,)
    )
    con.close()
    return df


# ── Appointment helpers ────────────────────────────────────────────────────────
def request_appointment(patient_id: str, doctor_id: int,
                        date: str, time: str, reason: str) -> int:
    return _write(
        """INSERT INTO appointments
           (patient_id,doctor_id,requested_date,requested_time,
            reason,status,created_at)
           VALUES (?,?,?,?,?,'pending',?)""",
        (patient_id, doctor_id, date, time, reason, datetime.now().isoformat()),
    )


def get_patient_appointments(patient_id: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT a.appt_id AS appointment_id,
               a.requested_date, a.requested_time, a.reason, a.status,
               a.doctor_notes, a.created_at,
               d.full_name AS doctor_name
        FROM appointments a
        LEFT JOIN doctors d ON a.doctor_id = d.doctor_id
        WHERE a.patient_id = ?
        ORDER BY a.requested_date DESC, a.requested_time DESC
    """, con, params=(patient_id,))
    con.close()
    return df


def get_doctor_appointments(doctor_id: int) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT a.appt_id AS appointment_id,
               a.requested_date, a.requested_time, a.reason, a.status,
               a.doctor_notes, a.created_at,
               p.full_name AS patient_name, p.phone, p.email
        FROM appointments a
        LEFT JOIN patients p ON a.patient_id = p.patient_id
        WHERE a.doctor_id = ?
        ORDER BY a.requested_date ASC, a.requested_time ASC
    """, con, params=(doctor_id,))
    con.close()
    return df


def update_appointment_status(appointment_id: int, status: str, notes: str = "") -> None:
    _write(
        "UPDATE appointments SET status=?, doctor_notes=? WHERE appt_id=?",
        (status, notes, appointment_id),
    )


def get_patient_followups(patient_id: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query("""
        SELECT v.visit_datetime, v.department, v.chief_complaint,
               v.urgency, v.severity_score, v.doctor_notes,
               d.full_name AS doctor_name
        FROM visits v
        LEFT JOIN doctors d ON v.doctor_id = d.doctor_id
        WHERE v.patient_id = ? AND v.follow_up_needed = 1
        ORDER BY v.visit_datetime DESC
    """, con, params=(patient_id,))
    con.close()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_whisper():
    try:
        return whisper.load_model("small")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_vision():
    try:
        return load_vision_engine()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_kvasir():
    try:
        return load_kvasir_engine()
    except Exception:
        return None


whisper_model  = _load_whisper()
vision_model   = _load_vision()
gradcam_engine = GradCAMPlusPlus(vision_model) if vision_model is not None else None
kvasir_model   = _load_kvasir()


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def audio_to_text(audio_file) -> Tuple[str, float]:
    audio_bytes = audio_file.read()
    data, sr    = sf.read(io.BytesIO(audio_bytes))
    data        = np.asarray(data, dtype=np.float32)
    result      = whisper_model.transcribe(data, fp16=False)
    return result["text"], len(data) / sr


def transcribe_file_whisper(audio_file) -> Optional[str]:
    if whisper_model is None:
        return None
    try:
        audio_bytes = (
            audio_file.read() if hasattr(audio_file, "read") else bytes(audio_file)
        )
        suffix = ".wav"
        if hasattr(audio_file, "name"):
            ext = os.path.splitext(audio_file.name)[1].lower()
            if ext in {".wav", ".mp3", ".m4a", ".ogg", ".mp4"}:
                suffix = ext
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        result = whisper_model.transcribe(tmp_path)
        os.unlink(tmp_path)
        return str(result.get("text", "")).strip()
    except Exception:
        return None


def get_image_tensor(img: Image.Image) -> torch.Tensor:
    tf_pipeline = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf_pipeline(img.convert("RGB")).unsqueeze(0).to(DEVICE)


def friendly_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").title()


def render_imaging_findings(findings_str: Optional[str]) -> None:
    if not findings_str:
        st.info("No findings available.")
        return
    try:
        data = json.loads(findings_str)
    except Exception:
        st.write(findings_str)
        return

    if isinstance(data, dict):
        if "risk_tier" in data:
            diag = str(data.get("class", "Unknown"))
            conf = float(data.get("confidence", 0.0)) * 100
            st.markdown(
                f"<div class='diagnosis-pill'><b>Condition:</b> {friendly_label(diag)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='accuracy-pill'><b>Confidence:</b> {conf:.1f}%</div>",
                unsafe_allow_html=True,
            )
            return

        numeric = {k: v for k, v in data.items() if isinstance(v, (int, float, np.floating, np.integer))}
        if numeric:
            top, top_conf = max(numeric.items(), key=lambda x: float(x[1]))
            if float(top_conf) > 0.40:
                st.markdown(
                    f"<div class='diagnosis-pill'><b>Primary Finding:</b> {friendly_label(top)}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='accuracy-pill'><b>Confidence:</b> {float(top_conf)*100:.1f}%</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='accuracy-pill'><b>Status:</b> Normal / No significant abnormalities detected.</div>",
                    unsafe_allow_html=True,
                )
            return
    st.write(data)


def render_longitudinal_chart(patient_id: str) -> None:
    records = _fetch_all(
        "SELECT timestamp, view_position, all_findings FROM medical_records WHERE patient_id=? ORDER BY timestamp ASC",
        (patient_id,),
    )
    points = []
    for rec in records:
        if not rec.get("all_findings"):
            continue
        try:
            findings = json.loads(rec["all_findings"])
        except Exception:
            continue
        if isinstance(findings, dict) and "risk_tier" not in findings:
            entry: Dict[str, Any] = {"Timestamp": rec["timestamp"], "Modality": rec["view_position"] or "Imaging"}
            for k, v in findings.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    entry[k] = float(v)
            points.append(entry)

    if not points:
        return
    df   = pd.DataFrame(points)
    paths = [c for c in df.columns if c not in ("Timestamp", "Modality")]
    if not paths:
        return
    sel = st.multiselect("Track pathologies:", paths, default=paths[:2])
    if sel:
        melt = df.melt(id_vars=["Timestamp", "Modality"], value_vars=sel,
                       var_name="Condition", value_name="Probability")
        fig  = px.line(melt, x="Timestamp", y="Probability",
                       color="Condition", markers=True, template="plotly_white")
        fig.update_layout(yaxis_title="AI Confidence")
        st.plotly_chart(fig, use_container_width=True)


def render_chat(sender_id: int, receiver_id: int, partner_name: str) -> None:
    st.markdown(f"### 💬 Secure Chat with {partner_name}")
    chat_box = st.container(height=400)
    msgs = _fetch_all(
        """SELECT sender_id, message_body, timestamp FROM chat_messages
           WHERE (sender_id=? AND receiver_id=?) OR (sender_id=? AND receiver_id=?)
           ORDER BY timestamp ASC""",
        (sender_id, receiver_id, receiver_id, sender_id),
    )
    with chat_box:
        if not msgs:
            st.info("No messages yet.")
        for m in msgs:
            role = "user" if m["sender_id"] == sender_id else "assistant"
            name = "You" if m["sender_id"] == sender_id else partner_name
            with st.chat_message(role):
                st.caption(f"{name} · {m['timestamp']}")
                st.write(m["message_body"])

    prompt = st.chat_input("Type a message…")
    if prompt and prompt.strip():
        _write(
            "INSERT INTO chat_messages (sender_id, receiver_id, message_body) VALUES (?,?,?)",
            (sender_id, receiver_id, prompt.strip()),
        )
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "doctor" not in st.session_state:
    st.session_state.doctor = None
if "patient" not in st.session_state:
    st.session_state.patient = None

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN / REGISTER SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.doctor is None and st.session_state.patient is None:
    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🏥 SehaTrack Pro")
        st.markdown("#### Clinical Intelligence Platform")
        st.markdown("---")

        who = st.radio("I am a:", ["Doctor", "Patient"], horizontal=True)

        if who == "Doctor":
            tab_login, tab_reg = st.tabs(["🔐 Doctor Login", "➕ Register"])

            with tab_login:
                username = st.text_input("Username", key="li_u")
                password = st.text_input("Password", type="password", key="li_p")
                if st.button("Log In", key="btn_dlogin"):
                    row = verify_login(username, password)
                    if row:
                        st.session_state.doctor = {
                            "id": row[0], "name": row[1],
                            "specialty": row[2], "username": username,
                        }
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

            with tab_reg:
                r_name = st.text_input("Full name (e.g. Dr. Sara Ali)")
                r_spec = st.selectbox("Specialty", [
                    "General Practice", "Internal Medicine", "Pulmonology",
                    "Cardiology", "Neurology", "Gastroenterology",
                    "Orthopaedics", "Emergency Medicine", "Paediatrics", "Other",
                ])
                r_user = st.text_input("Username", key="reg_u")
                r_pw   = st.text_input("Password", type="password", key="reg_p")
                r_pw2  = st.text_input("Confirm password", type="password", key="reg_p2")
                if st.button("Create Account", key="btn_dreg"):
                    if r_pw != r_pw2:
                        st.error("Passwords do not match.")
                    elif len(r_pw) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif not r_user or not r_name:
                        st.error("Please fill in all fields.")
                    elif register_doctor(r_user, r_pw, r_name, r_spec):
                        st.success("Account created! You can now log in.")
                    else:
                        st.error("Username already taken.")

        else:  # Patient
            tab_plogin, tab_preg = st.tabs(["🔐 Patient Login", "➕ Create Account"])

            with tab_plogin:
                pu = st.text_input("Username", key="pli_u")
                pp = st.text_input("Password", type="password", key="pli_p")
                if st.button("Log In", key="btn_plogin"):
                    pat = verify_patient_login(pu, pp)
                    if pat:
                        st.session_state.patient = pat
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

            with tab_preg:
                st.caption("Enter your full name and date of birth exactly as given to your doctor.")
                pr_name = st.text_input("Full name", key="preg_name", placeholder="e.g. Ahmed Khalil")
                pr_dob  = st.date_input("Date of birth", key="preg_dob",
                                        min_value=datetime(1900, 1, 1).date())

                if st.button("Find My Record", key="btn_find"):
                    derived_pid = _patient_id(pr_name.strip(), str(pr_dob))
                    existing    = _fetch_one(
                        "SELECT patient_id, full_name, date_of_birth, gender, age FROM patients WHERE patient_id=?",
                        (derived_pid,)
                    )
                    if existing:
                        st.session_state["found_patient"] = existing
                        st.success(f"✅ Record found: **{existing['full_name']}**, "
                                   f"DOB {existing['date_of_birth']}, "
                                   f"{existing.get('gender','—')}, age {existing.get('age','—')}")
                    else:
                        st.session_state["found_patient"] = None
                        # Show all patients whose name loosely matches for debugging
                        close = _fetch_all(
                            "SELECT full_name, date_of_birth, patient_id FROM patients WHERE full_name LIKE ?",
                            (f"%{pr_name.strip().split()[0]}%",)
                        ) if pr_name.strip() else []
                        st.error("No record found. Check your name and date of birth match what your doctor entered.")
                        if close:
                            st.warning("Did you mean one of these? Ask your doctor to confirm the spelling:")
                            for c in close:
                                st.caption(f"• {c['full_name']} — DOB: {c['date_of_birth']}")

                if st.session_state.get("found_patient"):
                    found = st.session_state["found_patient"]
                    if found.get("password_hash") if "password_hash" in found else _fetch_one(
                        "SELECT password_hash FROM patients WHERE patient_id=?", (found["patient_id"],)
                    ).get("password_hash"):
                        st.warning("This patient already has a portal account. Please log in instead.")
                    else:
                        st.markdown("---")
                        st.markdown("#### Set your login credentials")
                        pr_user = st.text_input("Choose a username", key="preg_u")
                        pr_pw   = st.text_input("Choose a password", type="password", key="preg_p")
                        pr_pw2  = st.text_input("Confirm password",  type="password", key="preg_p2")
                        if st.button("Create Account", key="btn_preg"):
                            if pr_pw != pr_pw2:
                                st.error("Passwords do not match.")
                            elif len(pr_pw) < 6:
                                st.error("Password must be at least 6 characters.")
                            elif not pr_user:
                                st.error("Please choose a username.")
                            elif register_patient_account(found["patient_id"], pr_user, pr_pw):
                                st.success("✅ Account created! You can now log in.")
                                st.session_state["found_patient"] = None
                            else:
                                st.error("Username already taken. Please choose another.")

               
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT PORTAL
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.patient is not None:
    pat = st.session_state.patient
    pid = pat["patient_id"]

    with st.sidebar:
        st.markdown(
            f"<div class='patient-badge'>🧑 {pat['full_name']}</div>",
            unsafe_allow_html=True,
        )
        st.caption("Patient Portal")
        st.markdown("---")
        pat_page = st.radio("MENU", [
            "💊 My Prescriptions",
            "📅 Book Appointment",
            "🔔 Follow-up Reminders",
            "📋 My Visits",
            "🩻 My Imaging",
            "💬 Chat with Doctor",
        ])
        st.markdown("---")
        if st.button("🚪 Log Out"):
            st.session_state.patient = None
            st.rerun()

    # ── My Prescriptions ─────────────────────────────────────────────────────
    if pat_page == "💊 My Prescriptions":
        st.header("💊 My Prescriptions")
        rx = get_patient_prescriptions(pid)
        if rx.empty:
            st.info("No prescriptions on record yet. They will appear here after your doctor issues them.")
        else:
            st.caption(f"{len(rx)} prescription(s) on file")
            for _, r in rx.iterrows():
                with st.expander(
                    f"💊 {r['medication_name']}  ·  {r.get('dosage','—')}  ·  issued {str(r.get('issued_at',''))[:10]}"
                ):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Medication:** {r['medication_name']}")
                        st.write(f"**Dosage:** {r.get('dosage') or '—'}")
                        st.write(f"**Frequency:** {r.get('frequency') or '—'}")
                    with c2:
                        st.write(f"**Duration:** {r.get('duration') or '—'}")
                        st.write(f"**Prescribed by:** {r.get('doctor_name') or '—'}")
                        st.write(f"**Visit date:** {str(r.get('visit_datetime','—'))[:10]}")
                    if r.get("notes"):
                        st.info(f"📝 {r['notes']}")

    # ── Book Appointment ──────────────────────────────────────────────────────
    elif pat_page == "📅 Book Appointment":
        st.header("📅 Request an Appointment")

        linked_docs = _fetch_all(
            """SELECT DISTINCT d.doctor_id, d.full_name, d.specialty
               FROM visits v JOIN doctors d ON v.doctor_id = d.doctor_id
               WHERE v.patient_id = ?""",
            (pid,),
        )
        if not linked_docs:
            st.info("No doctor linked to your account yet. You need at least one visit on record before booking.")
        else:
            doc_opts = {f"{v['full_name']} — {v['specialty']}": v for v in linked_docs}
            sel_doc_str = st.selectbox("Select doctor:", list(doc_opts.keys()))
            sel_doc     = doc_opts[sel_doc_str]

            req_date = st.date_input("Preferred date", min_value=datetime.now().date())
            req_time = st.selectbox("Preferred time", [
                "08:00", "08:30", "09:00", "09:30", "10:00", "10:30",
                "11:00", "11:30", "12:00", "14:00", "14:30", "15:00",
                "15:30", "16:00", "16:30", "17:00",
            ])
            reason = st.text_area("Reason for visit", placeholder="Describe your symptoms or concern…")

            if st.button("📨 Send Request"):
                if not reason.strip():
                    st.error("Please describe the reason for your visit.")
                else:
                    request_appointment(
                        pid, sel_doc["doctor_id"],
                        str(req_date), req_time, reason.strip()
                    )
                    st.success("✅ Request sent! Your doctor will confirm shortly.")
                    st.rerun()

            # Show existing requests
            st.markdown("---")
            st.markdown("#### My Appointment Requests")
            appts = get_patient_appointments(pid)
            if appts.empty:
                st.info("No appointment requests yet.")
            else:
                STATUS_ICON  = {"pending": "🟡", "approved": "🟢", "rejected": "🔴", "completed": "⚪"}
                STATUS_CLASS = {"pending": "appt-pending", "approved": "appt-approved",
                                "rejected": "appt-rejected", "completed": "section-card"}
                for _, a in appts.iterrows():
                    icon  = STATUS_ICON.get(a["status"], "❓")
                    css   = STATUS_CLASS.get(a["status"], "section-card")
                    notes = f"<br><small>📝 {a['doctor_notes']}</small>" if a.get("doctor_notes") else ""
                    st.markdown(
                        f"<div class='{css}'>"
                        f"{icon} <b>{a['requested_date']} at {a['requested_time']}</b> "
                        f"with {a['doctor_name']} &nbsp;·&nbsp; {a['status'].capitalize()}<br>"
                        f"<small>{a['reason']}</small>{notes}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── Follow-up Reminders ───────────────────────────────────────────────────
    elif pat_page == "🔔 Follow-up Reminders":
        st.header("🔔 Follow-up Reminders")
        followups = get_patient_followups(pid)

        if followups.empty:
            st.success("✅ No pending follow-ups. You're all clear!")
        else:
            st.warning(f"You have **{len(followups)}** visit(s) that require a follow-up.")
            for _, f in followups.iterrows():
                with st.expander(
                    f"🔔 {str(f['visit_datetime'])[:10]} — {f['department']} · {str(f['chief_complaint'])[:50]}"
                ):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Visit date:** {str(f['visit_datetime'])[:10]}")
                        st.write(f"**Department:** {f['department']}")
                        st.write(f"**Complaint:** {f['chief_complaint']}")
                    with c2:
                        st.write(f"**Urgency:** {f['urgency']}")
                        st.write(f"**Severity:** {f['severity_score']}/10")
                        st.write(f"**Doctor:** {f.get('doctor_name') or '—'}")
                    if f.get("doctor_notes"):
                        st.info(f"📝 Doctor's note: {f['doctor_notes']}")
            st.markdown("---")
            st.info("💡 Use the **Book Appointment** page to schedule your follow-up.")

    # ── My Visits ─────────────────────────────────────────────────────────────
    elif pat_page == "📋 My Visits":
        st.header("📋 My Visit History")
        hist = get_patient_history(pid)
        if hist.empty:
            st.info("No visits recorded yet.")
        else:
            disp = hist[[
                "visit_datetime", "visit_type", "department",
                "chief_complaint", "symptom_top1", "urgency",
                "severity_score", "follow_up_needed", "doctor_name", "doctor_notes",
            ]].copy()
            disp.columns = [
                "Date", "Type", "Department", "Complaint",
                "Primary Symptom", "Urgency", "Severity",
                "Follow-up", "Doctor", "Doctor Notes",
            ]
            st.dataframe(disp, use_container_width=True)

    # ── My Imaging ────────────────────────────────────────────────────────────
    elif pat_page == "🩻 My Imaging":
        st.header("🩻 My Imaging Records")
        recs = _fetch_all(
            "SELECT * FROM medical_records WHERE patient_id=? ORDER BY timestamp DESC",
            (pid,),
        )
        if not recs:
            st.info("No imaging records yet.")
        for r in recs:
            with st.expander(f"📷 {r['timestamp']} — {r.get('view_position', '?')}"):
                if r.get("image_path") and os.path.isfile(r["image_path"]):
                    st.image(r["image_path"], width=300)
                render_imaging_findings(r.get("all_findings"))

    # ── Chat with Doctor ──────────────────────────────────────────────────────
    elif pat_page == "💬 Chat with Doctor":
        st.header("💬 Chat with Your Doctor")
        visits = _fetch_all(
            """SELECT DISTINCT d.doctor_id, d.full_name, d.specialty
               FROM visits v JOIN doctors d ON v.doctor_id = d.doctor_id
               WHERE v.patient_id = ?""",
            (pid,),
        )
        if not visits:
            st.info("No doctor linked yet. You need at least one visit on record.")
        else:
            doc_options = {f"{v['full_name']} ({v['specialty']})": v for v in visits}
            sel_doc_str = st.selectbox("Select your doctor:", list(doc_options.keys()))
            sel_doc     = doc_options[sel_doc_str]
            render_chat(
                _patient_chat_id(pid),
                sel_doc["doctor_id"],
                sel_doc["full_name"],
            )

    st.stop()  # Don't fall through to the doctor UI


# ══════════════════════════════════════════════════════════════════════════════
# DOCTOR SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
doc = st.session_state.doctor
with st.sidebar:
    st.markdown(f"<div class='doctor-badge'>👨‍⚕️ {doc['name']}</div>", unsafe_allow_html=True)
    st.caption(doc["specialty"])
    st.markdown("---")
    page = st.radio("MENU", [
        "🩺 Diagnostic Lab",
        "🩻 Imaging Analysis",
        "📅 Appointments",
        "💬 Tele-Chat",
        "📊 Insights Hub",
        "📋 Clinical Logs",
        "👤 Patient Search",
    ])
    st.markdown("---")
    if st.button("🚪 Log Out"):
        st.session_state.doctor = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DIAGNOSTIC LAB
# ══════════════════════════════════════════════════════════════════════════════
if page == "🩺 Diagnostic Lab":
    st.header("🩺 Diagnostic Lab")
    st.caption(f"Logged in as **{doc['name']}** · {doc['specialty']}")

    st.markdown("### 1 — Patient Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        p_name  = st.text_input("Full name *", placeholder="e.g. Ahmed Khalil")
        p_dob   = st.date_input("Date of birth *", min_value=datetime(1900, 1, 1).date())
        p_phone = st.text_input("Phone number", placeholder="+20 1xx xxx xxxx")
    with c2:
        p_gender    = st.selectbox("Gender *", ["Male", "Female", "Other"])
        p_blood     = st.selectbox("Blood type", ["Unknown", "A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
        p_insurance = st.selectbox("Insurance status", ["Insured", "Uninsured", "Government"])
    with c3:
        p_email     = st.text_input("Email", placeholder="patient@email.com")
        p_emergency = st.text_input("Emergency contact", placeholder="Name — phone")
        p_allergies = st.text_area("Known allergies", placeholder="e.g. Penicillin, Aspirin", height=68)

    age = int((datetime.now().date() - p_dob).days / 365.25)

    st.markdown("### 2 — Visit Details")
    v1, v2, v3 = st.columns(3)
    with v1:
        visit_type = st.selectbox("Visit type", ["Walk-in", "Scheduled", "Emergency", "Telehealth"])
    with v2:
        department = st.selectbox("Department", [
            "General Practice", "Internal Medicine", "Pulmonology",
            "Cardiology", "Neurology", "Gastroenterology",
            "Orthopaedics", "Emergency Medicine", "Paediatrics",
        ])
    with v3:
        severity = st.slider("Severity (1–10)", 1, 10, 5)

    follow_up    = st.checkbox("Follow-up required")
    doctor_notes = st.text_area("Doctor notes", placeholder="Clinical observations…", height=80)

    st.markdown("### 3 — Symptom Input")
    audio_input = st.audio_input("🎙️ Record symptoms (optional)")
    user_text   = st.text_input("Or type symptoms manually",
                                placeholder="e.g. persistent cough and fever for 3 days")

    if st.button("🔍 Analyze & Save to Database"):
        if not p_name:
            st.error("Patient full name is required.")
            st.stop()

        with st.spinner("Running AI pipeline…"):
            transcription  = ""
            audio_duration = None

            if audio_input:
                try:
                    transcription, audio_duration = audio_to_text(audio_input)
                    st.info(f"🎙️ Transcribed: *{transcription}*")
                    text_to_analyze = transcription
                except Exception as e:
                    st.warning(f"Audio transcription failed: {e}")
                    text_to_analyze = user_text or "fever and cough"
            else:
                text_to_analyze = user_text or "fever and cough"
                transcription   = text_to_analyze

            try:
                predictions = predict_topk(text_to_analyze)
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                st.stop()

            top1 = predictions[0] if predictions else {"label": "unknown", "score": 0}
            top2 = predictions[1] if len(predictions) > 1 else {"label": None, "score": None}
            top3 = predictions[2] if len(predictions) > 2 else {"label": None, "score": None}

            urgency_map = {range(1, 4): "routine", range(4, 6): "soon",
                           range(6, 8): "urgent", range(8, 11): "emergency"}
            urgency = next((v for k, v in urgency_map.items() if severity in k), "routine")

            patient_data = {
                "full_name": p_name, "date_of_birth": str(p_dob), "age": age,
                "gender": p_gender, "blood_type": p_blood, "phone": p_phone,
                "email": p_email, "insurance_status": p_insurance,
                "emergency_contact": p_emergency, "allergies": p_allergies,
            }
            patient_id, is_returning = get_or_create_patient(patient_data)

            vd = {
                "visit_type": visit_type, "department": department,
                "chief_complaint": text_to_analyze, "transcription": transcription,
                "symptom_top1": top1["label"], "symptom_top1_score": top1["score"],
                "symptom_top2": top2["label"], "symptom_top2_score": top2["score"],
                "symptom_top3": top3["label"], "symptom_top3_score": top3["score"],
                "urgency": urgency, "severity_score": severity,
                "follow_up_needed": follow_up, "doctor_notes": doctor_notes,
                "recommended_steps": "", "audio_duration_sec": audio_duration,
            }
            visit_id = save_visit(patient_id, doc["id"], vd, [])

        if is_returning:
            st.info(
                f"🔁 Returning patient — {len(get_patient_history(patient_id))} "
                f"previous visit(s). New visit added."
            )
        else:
            st.success("✅ New patient registered and visit saved.")
        st.caption(f"Patient ID: `{patient_id}` · Visit ID: `{visit_id}`")

        if transcription:
            st.markdown("#### 🎙️ Transcription")
            st.info(f'"{transcription}"')
            if audio_duration:
                st.caption(f"Audio duration: {audio_duration:.1f}s")

        URGENCY_ICON = {"routine": "🟢", "soon": "🟡", "urgent": "🟠", "emergency": "🔴"}
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Top Symptom",  top1["label"])
        r2.metric("Confidence",   f"{top1['score']:.1%}")
        r3.metric("Severity",     f"{severity}/10")
        r4.metric("Urgency", f"{URGENCY_ICON.get(urgency, '')} {urgency.capitalize()}")

        st.markdown("#### 🩺 All Predicted Symptoms")
        pred_df        = pd.DataFrame(predictions)
        pred_df["pct"] = (pred_df["score"] * 100).round(2)
        colors         = ["#2563eb" if i == 0 else "#93c5fd" for i in range(len(pred_df))]
        fig_preds = go.Figure(go.Bar(
            x=pred_df["pct"], y=pred_df["label"], orientation="h",
            marker_color=colors,
            text=pred_df["pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        fig_preds.update_layout(
            xaxis=dict(title="Confidence (%)", range=[0, max(pred_df["pct"]) * 1.15]),
            yaxis=dict(autorange="reversed"),
            height=max(300, len(pred_df) * 32),
            margin=dict(l=10, r=40, t=10, b=30),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_preds, use_container_width=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=top1["score"] * 100,
            title={"text": f"Confidence — {top1['label']}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 50],   "color": "#fee2e2"},
                    {"range": [50, 75],  "color": "#fef9c3"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ],
            },
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Prescriptions section (shown after visit is saved) ────────────────
        st.markdown("### 4 — Prescriptions")
        with st.expander("➕ Add medications to this visit"):
            num_meds = st.number_input("Number of medications", 1, 10, 1, key="num_meds")
            meds_to_save = []
            for i in range(int(num_meds)):
                st.markdown(f"**Medication {i + 1}**")
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    mname = st.text_input("Name",      key=f"mname_{i}", placeholder="e.g. Amoxicillin")
                with mc2:
                    mdose = st.text_input("Dosage",    key=f"mdose_{i}", placeholder="e.g. 500mg")
                with mc3:
                    mfreq = st.text_input("Frequency", key=f"mfreq_{i}", placeholder="e.g. 3x daily")
                with mc4:
                    mdur  = st.text_input("Duration",  key=f"mdur_{i}",  placeholder="e.g. 7 days")
                mnotes = st.text_input("Notes", key=f"mnotes_{i}", placeholder="Take after meals")
                if mname:
                    meds_to_save.append({
                        "medication_name": mname, "dosage": mdose,
                        "frequency": mfreq, "duration": mdur, "notes": mnotes,
                    })

            if st.button("💊 Save Prescriptions", key="btn_rx"):
                if meds_to_save:
                    add_prescription(visit_id, patient_id, doc["id"], meds_to_save)
                    st.success(f"Saved {len(meds_to_save)} prescription(s).")
                else:
                    st.warning("Enter at least one medication name.")

        st.markdown(f"#### 📋 Medical history — {p_name}")
        hist = get_patient_history(patient_id)
        if not hist.empty:
            disp = hist[[
                "visit_datetime", "visit_type", "department", "chief_complaint",
                "asr_transcription", "symptom_top1", "symptom_top1_score",
                "symptom_top2", "symptom_top3", "urgency", "severity_score",
                "follow_up_needed", "doctor_name", "doctor_notes",
            ]].copy()
            disp.columns = [
                "Date", "Type", "Dept", "Chief Complaint", "Transcription",
                "Symptom 1", "Confidence", "Symptom 2", "Symptom 3",
                "Urgency", "Severity", "Follow-up", "Doctor", "Notes",
            ]
            disp["Confidence"] = disp["Confidence"].apply(
                lambda v: f"{v:.1%}" if pd.notna(v) else ""
            )
            st.dataframe(disp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — IMAGING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🩻 Imaging Analysis":
    st.header("🩻 Imaging Analysis")
    st.caption("CheXNet chest X-ray analysis · Kvasir GI endoscopy · Grad-CAM++ · LIME")

    patients_raw = _fetch_all("SELECT patient_id, full_name, age, gender FROM patients ORDER BY full_name")
    if not patients_raw:
        st.info("No patients in the database yet. Register a patient in Diagnostic Lab first.")
        st.stop()

    patient_options = {f"{p['full_name']} ({p['patient_id']})": p for p in patients_raw}
    sel_pat_str     = st.selectbox("Select patient:", list(patient_options.keys()))
    sel_pat         = patient_options[sel_pat_str]
    pid             = sel_pat["patient_id"]

    analysis_type = st.selectbox(
        "Select modality:",
        ["Chest X-Ray — CheXNet + Grad-CAM++", "GI Endoscopy — Kvasir + LIME"],
    )
    doc_upload = st.file_uploader("Upload scan image:", type=["jpg", "jpeg", "png"])

    if doc_upload and st.button("🚀 Run Imaging Analysis"):
        img = Image.open(doc_upload).convert("RGB")

        if "Chest" in analysis_type:
            if vision_model is None:
                st.error("CheXNet model not available. Place best_chexnet_multimodal.pth next to app.py.")
                st.stop()

            with st.spinner("Running CheXNet multimodal inference + Grad-CAM++…"):
                img_tensor  = get_image_tensor(img)
                p_age       = sel_pat.get("age") or 35
                p_gender    = sel_pat.get("gender") or "Male"
                meta_tensor = encode_meta(p_age, p_gender, "PA/AP").to(DEVICE)

                with torch.no_grad():
                    logits = vision_model(img_tensor, meta_tensor)
                    preds  = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()

                res_map        = {DISEASE_LABELS[i]: float(preds[i]) for i in range(min(len(DISEASE_LABELS), len(preds)))}
                detected_paths = [
                    f"{lbl} ({float(preds[i]):.1%})"
                    for i, lbl in enumerate(DISEASE_LABELS)
                    if i < len(preds) and float(preds[i]) >= OPTIMAL_THRESHOLDS.get(lbl, 0.15)
                ]
                primary_dx = ", ".join(detected_paths) if detected_paths else "Normal / No Findings"

            st.success(f"**Detected:** {primary_dx}")

            if gradcam_engine is not None:
                try:
                    target_idx = int(np.argmax(preds))
                    heatmap    = gradcam_engine.generate(img_tensor, meta_tensor, target_idx)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, caption="Original X-Ray", use_container_width=True)
                    with col2:
                        if heatmap is not None:
                            overlay = GradCAMPlusPlus.overlay(img, heatmap, alpha=0.40)
                            st.image(overlay,
                                     caption=f"Grad-CAM++ · {DISEASE_LABELS[target_idx]}",
                                     use_container_width=True)
                        else:
                            st.warning("Grad-CAM++ could not generate a heatmap.")
                except Exception as e:
                    st.warning(f"Grad-CAM++ error: {e}")
                    st.image(img, caption="Original X-Ray", use_container_width=True)

            st.markdown("#### All Pathology Probabilities")
            prob_df = pd.DataFrame(
                [(lbl, float(preds[i]) * 100)
                 for i, lbl in enumerate(DISEASE_LABELS) if i < len(preds)],
                columns=["Pathology", "Probability (%)"],
            ).sort_values("Probability (%)", ascending=True)
            fig = px.bar(prob_df, x="Probability (%)", y="Pathology",
                         orientation="h", color="Probability (%)",
                         color_continuous_scale="Blues", template="plotly_white")
            fig.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Threshold")
            fig.update_layout(height=500, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            os.makedirs(os.path.join(_HERE, "saved"), exist_ok=True)
            save_path = os.path.join(_HERE, "saved", f"dx_{pid}_{int(datetime.now().timestamp())}.png")
            img.save(save_path)
            _write(
                """INSERT INTO medical_records
                   (patient_id,doctor_id,image_path,view_position,all_findings,timestamp)
                   VALUES (?,?,?,?,?,?)""",
                (pid, doc["id"], save_path, "PA/AP",
                 json.dumps(res_map), datetime.now().strftime("%Y-%m-%d %H:%M")),
            )
            st.caption("Record saved to medical history.")

        else:
            if kvasir_model is None or not _TF_AVAILABLE:
                st.error("Kvasir model not available. Place gi_model_clean.h5 next to app.py.")
                st.stop()

            with st.spinner("Running Kvasir GI inference + LIME explanation…"):
                class_name, confidence, chart_data, lime_img = run_kvasir_lime_explanation(
                    img, kvasir_model, num_samples=500
                )

            st.success(f"**Finding:** {friendly_label(class_name)} ({confidence:.1%} confidence)")

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original Endoscopy Frame", use_container_width=True)
            with col2:
                st.image(lime_img, caption="LIME Positive Feature Segments", use_container_width=True)

            if chart_data:
                st.markdown("#### LIME Attribution Weights")
                st.dataframe(pd.DataFrame(chart_data), use_container_width=True)

            res_map   = {"class": class_name, "confidence": confidence, "risk_tier": "evaluated"}
            os.makedirs(os.path.join(_HERE, "saved"), exist_ok=True)
            save_path = os.path.join(_HERE, "saved", f"gi_{pid}_{int(datetime.now().timestamp())}.png")
            img.save(save_path)
            _write(
                """INSERT INTO medical_records
                   (patient_id,doctor_id,image_path,view_position,all_findings,timestamp)
                   VALUES (?,?,?,?,?,?)""",
                (pid, doc["id"], save_path, "Endoscopy",
                 json.dumps(res_map), datetime.now().strftime("%Y-%m-%d %H:%M")),
            )
            st.caption("Record saved to medical history.")

    st.markdown("---")
    st.markdown(f"#### 📂 Imaging History — {sel_pat['full_name']}")
    recs = _fetch_all(
        "SELECT * FROM medical_records WHERE patient_id=? ORDER BY timestamp DESC", (pid,)
    )
    if not recs:
        st.info("No imaging records yet for this patient.")
    for r in recs:
        with st.expander(f"Record #{r['record_id']} — {r['timestamp']} ({r.get('view_position','?')})"):
            if r.get("image_path") and os.path.isfile(r["image_path"]):
                st.image(r["image_path"], width=280)
            render_imaging_findings(r.get("all_findings"))
            if r.get("doctor_notes"):
                st.info(f"**Doctor Notes:** {r['doctor_notes']}")

    st.markdown("#### 📈 Longitudinal Pathology Tracking")
    render_longitudinal_chart(pid)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — APPOINTMENTS (Doctor side)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Appointments":
    st.header("📅 Appointment Requests")

    appts = get_doctor_appointments(doc["id"])
    if appts.empty:
        st.info("No appointment requests yet.")
        st.stop()

    STATUS_COLOR = {"pending": "🟡", "approved": "🟢", "rejected": "🔴", "completed": "⚪"}

    pending = appts[appts["status"] == "pending"]
    rest    = appts[appts["status"] != "pending"]

    if not pending.empty:
        st.markdown(f"### 🟡 Pending ({len(pending)})")
        print(appts.columns)
        for _, row in pending.iterrows():
            with st.expander(
                f"{row['patient_name']} — {row['requested_date']} {row['requested_time']} · {str(row['reason'])[:60]}"
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Patient:** {row['patient_name']}")
                    st.write(f"**Requested:** {row['requested_date']} at {row['requested_time']}")
                    st.write(f"**Reason:** {row['reason']}")
                with c2:
                    st.write(f"**Phone:** {row.get('phone') or '—'}")
                    st.write(f"**Email:** {row.get('email') or '—'}")
                    st.write(f"**Submitted:** {str(row.get('created_at',''))[:10]}")

                doc_note = st.text_input(
                    "Note to patient (optional)",
                    key=f"anote_{row['appointment_id']}"
                )

                ca, cb, cc = st.columns(3)
                with ca:
                    if st.button("✅ Confirm", key=f"conf_{row['appointment_id']}"):
                        update_appointment_status(row["appointment_id"], "approved", doc_note)
                        st.rerun()
                with cb:
                    if st.button("❌ Reject", key=f"rej_{row['appointment_id']}"):
                        update_appointment_status(row["appointment_id"], "rejected", doc_note)
                        st.rerun()
                with cc:
                    if st.button("✔️ Mark Completed", key=f"comp_{row['appointment_id']}"):
                        update_appointment_status(row["appointment_id"], "completed", doc_note)
                        st.rerun()

    else:
        st.success("No pending requests right now.")

    if not rest.empty:
        st.markdown("---")
        st.markdown("### Past / Resolved")
        display = rest[[
            "requested_date", "requested_time", "patient_name",
            "reason", "status", "doctor_notes"
        ]].copy()
        display["status"] = display["status"].apply(
            lambda s: f"{STATUS_COLOR.get(s, '')} {s.capitalize()}"
        )
        display.columns = ["Date", "Time", "Patient", "Reason", "Status", "Your Note"]
        st.dataframe(display, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TELE-CHAT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Tele-Chat":
    st.header("💬 Tele-Chat")
    st.caption("Secure messaging between doctor and patient")

    chat_patients = _fetch_all(
        """SELECT DISTINCT p.patient_id, p.full_name
           FROM visits v JOIN patients p ON v.patient_id = p.patient_id
           WHERE v.doctor_id = ?""",
        (doc["id"],),
    )
    if not chat_patients:
        st.info("No patients linked to your account yet. Log visits in the Diagnostic Lab first.")
    else:
        options    = {p["full_name"]: p["patient_id"] for p in chat_patients}
        sel_name   = st.selectbox("Select patient:", list(options.keys()))
        partner_id = options[sel_name]
        render_chat(doc["id"], _patient_chat_id(partner_id), sel_name)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — INSIGHTS HUB
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Insights Hub":
    st.title("📊 Clinical Intelligence Dashboard")

    df = get_all_visits()
    if df.empty:
        st.warning("No visit data yet. Run some diagnoses first.")
        st.stop()

    df["visit_datetime"] = pd.to_datetime(df["visit_datetime"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Patients",    df["patient_id"].nunique())
    m2.metric("Total Visits",      len(df))
    m3.metric("Avg Severity",      f"{df['severity_score'].mean():.1f}")
    m4.metric("Follow-ups Needed", int(df["follow_up_needed"].sum()))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Urgency Distribution")
        fig_u = px.pie(
            df, names="urgency", hole=0.5, color="urgency",
            color_discrete_map={"routine": "#22C55E", "soon": "#F59E0B",
                                 "urgent": "#EF4444", "emergency": "#7C3AED"},
        )
        st.plotly_chart(fig_u, use_container_width=True)
    with c2:
        st.markdown("#### Top Predicted Symptoms")
        s_cnt = df["symptom_top1"].value_counts().head(10).reset_index()
        s_cnt.columns = ["symptom", "count"]
        fig_s = px.bar(s_cnt.sort_values("count"), x="count", y="symptom",
                       orientation="h", color="count", color_continuous_scale="Blues")
        fig_s.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("#### Weekly Visit Volume")
    weekly = df.set_index("visit_datetime").resample("W")["visit_id"].count().reset_index()
    weekly.columns = ["week", "visits"]
    fig_v = px.area(weekly, x="week", y="visits", color_discrete_sequence=["#3b82f6"])
    st.plotly_chart(fig_v, use_container_width=True)

    st.markdown("#### Patient Risk Matrix — Age vs Severity")
    fig_r = px.scatter(
        df, x="age", y="severity_score",
        color="urgency", size="severity_score", hover_name="full_name",
        color_discrete_map={"routine": "#22C55E", "soon": "#F59E0B",
                             "urgent": "#EF4444", "emergency": "#7C3AED"},
        template="plotly_white",
    )
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("#### Severity Distribution per Symptom")
    top_symp = df["symptom_top1"].value_counts().head(8).index
    fig_b    = px.box(df[df["symptom_top1"].isin(top_symp)],
                      x="symptom_top1", y="severity_score", color="symptom_top1")
    fig_b.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_b, use_container_width=True)

    # Appointment stats
    all_appts = get_doctor_appointments(doc["id"])
    if not all_appts.empty:
        st.markdown("#### Appointment Status Breakdown")
        appt_counts = all_appts["status"].value_counts().reset_index()
        appt_counts.columns = ["Status", "Count"]
        fig_a = px.pie(appt_counts, names="Status", values="Count", hole=0.4,
                       color="Status",
                       color_discrete_map={"pending": "#F59E0B", "approved": "#22C55E",
                                           "rejected": "#EF4444", "completed": "#94A3B8"})
        st.plotly_chart(fig_a, use_container_width=True)

    # Imaging stats
    records = _fetch_all("SELECT view_position, all_findings FROM medical_records")
    if records:
        st.markdown("#### AI Imaging Diagnoses")
        diagnoses = []
        for rec in records:
            if not rec.get("all_findings"):
                continue
            try:
                f = json.loads(rec["all_findings"])
            except Exception:
                continue
            if isinstance(f, dict) and "risk_tier" in f:
                diagnoses.append(friendly_label(str(f.get("class", "Unknown"))))
            elif isinstance(f, dict):
                numeric = {k: v for k, v in f.items() if isinstance(v, (int, float, np.floating, np.integer))}
                if numeric:
                    top, conf = max(numeric.items(), key=lambda x: float(x[1]))
                    if float(conf) > 0.40:
                        diagnoses.append(friendly_label(top))

        if diagnoses:
            dx_df  = pd.DataFrame({"Condition": diagnoses})
            counts = dx_df["Condition"].value_counts().reset_index()
            counts.columns = ["Condition", "count"]
            fig_dx = px.bar(counts, x="Condition", y="count",
                            title="Frequency of Imaging Diagnoses",
                            color="count", color_continuous_scale="Blues")
            st.plotly_chart(fig_dx, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — CLINICAL LOGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Clinical Logs":
    st.header("📋 Clinical Logs")

    df = get_all_visits()
    if df.empty:
        st.warning("No records found.")
        st.stop()

    df["visit_datetime"] = pd.to_datetime(df["visit_datetime"])

    f1, f2, f3 = st.columns(3)
    with f1:
        urgency_filter = st.multiselect("Urgency", df["urgency"].dropna().unique().tolist(),
                                        default=df["urgency"].dropna().unique().tolist())
    with f2:
        dept_filter = st.multiselect("Department", df["department"].dropna().unique().tolist(),
                                     default=df["department"].dropna().unique().tolist())
    with f3:
        date_range = st.date_input("Date range", [
            df["visit_datetime"].min().date(), df["visit_datetime"].max().date()
        ])

    filtered = df[
        df["urgency"].isin(urgency_filter) &
        df["department"].isin(dept_filter) &
        (df["visit_datetime"].dt.date >= date_range[0]) &
        (df["visit_datetime"].dt.date <= date_range[1])
    ] if len(date_range) == 2 else df

    st.caption(f"Showing {len(filtered)} records")
    st.dataframe(
        filtered[[
            "visit_datetime", "full_name", "age", "gender",
            "chief_complaint", "symptom_top1", "urgency",
            "severity_score", "follow_up_needed", "doctor_name", "department",
        ]],
        use_container_width=True,
    )
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export CSV", csv, "clinical_logs.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PATIENT SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Patient Search":
    st.header("👤 Patient Search")
    search = st.text_input("Search by patient name", placeholder="e.g. Ahmed")

    if search:
        con      = sqlite3.connect(DB_PATH)
        patients = pd.read_sql_query(
            "SELECT * FROM patients WHERE full_name LIKE ?",
            con, params=(f"%{search}%",)
        )
        con.close()

        if patients.empty:
            st.info("No patients found.")
        else:
            for _, p in patients.iterrows():
                has_account = bool(p.get("username"))
                acct_badge  = "🔐 Has portal account" if has_account else "⚪ No portal account"
                with st.expander(
                    f"🧑 {p['full_name']} — {p.get('gender','?')}, age {p.get('age','?')} "
                    f"| {p.get('blood_type','?')} | {p.get('insurance_status','?')} · {acct_badge}"
                ):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.write(f"**Patient ID:** `{p['patient_id']}`")
                        st.write(f"**DOB:** {p.get('date_of_birth','—')}")
                        st.write(f"**Phone:** {p.get('phone','—')}")
                        st.write(f"**Email:** {p.get('email','—')}")
                    with d2:
                        st.write(f"**Emergency contact:** {p.get('emergency_contact','—')}")
                        st.write(f"**Allergies:** {p.get('allergies','—')}")
                        st.write(f"**Registered:** {str(p.get('created_at','—'))[:10]}")
                        st.write(f"**Portal account:** {acct_badge}")

                    st.markdown("**Visit history:**")
                    hist = get_patient_history(p["patient_id"])
                    if hist.empty:
                        st.caption("No visits recorded yet.")
                    else:
                        st.dataframe(
                            hist[[
                                "visit_datetime", "chief_complaint", "symptom_top1",
                                "urgency", "severity_score", "doctor_name", "doctor_notes",
                            ]],
                            use_container_width=True,
                        )

                    st.markdown("**Prescriptions:**")
                    rx = get_patient_prescriptions(p["patient_id"])
                    if rx.empty:
                        st.caption("No prescriptions on file.")
                    else:
                        st.dataframe(
                            rx[["medication_name", "dosage", "frequency",
                                "duration", "issued_at", "doctor_name"]],
                            use_container_width=True,
                        )

                    st.markdown("**Imaging records:**")
                    img_recs = _fetch_all(
                        "SELECT * FROM medical_records WHERE patient_id=? ORDER BY timestamp DESC",
                        (p["patient_id"],),
                    )
                    if img_recs:
                        for r in img_recs:
                            st.caption(
                                f"📷 {r['timestamp']} — {r.get('view_position','?')}"
                                f"{'  ·  ' + r['doctor_notes'] if r.get('doctor_notes') else ''}"
                            )
                    else:
                        st.caption("No imaging records.")