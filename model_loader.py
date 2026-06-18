"""
model_loader.py — SehaTrack Pro
Centralised lazy-loading with memory-safe caching.

WHY THIS FILE EXISTS:
- Streamlit re-runs the entire script on every user interaction.
- Without @st.cache_resource, all 3 models would reload on EVERY click.
- This file wraps every model in a cached loader so they load ONCE and
  stay in memory across all user sessions.
- Each loader returns None if the weight file is missing — the app stays
  alive and shows a friendly "model not available" message instead of crashing.
"""

import os
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# WHISPER (speech-to-text)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading Whisper speech model…")
def get_whisper():
    """
    Loads the Whisper 'small' model (~244 MB download on first run).
    Downloaded automatically to ~/.cache/whisper/ — no manual step needed.
    Returns None if whisper is not installed.
    """
    try:
        import whisper
        return whisper.load_model("small")
    except Exception as e:
        st.warning(f"⚠️ Whisper not available: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# NLP SYMPTOM CLASSIFIER (HuggingFace)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading NLP symptom model…")
def get_nlp():
    """
    Loads the HuggingFace NLP classifier from the local 'model_only/' folder.
    Returns (tokenizer, model) or (None, None) if the folder is missing.
    """
    from model import NLP_MODEL_PATH
    import os, json
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not os.path.isdir(NLP_MODEL_PATH):
        st.warning(f"⚠️ NLP model folder not found at: {NLP_MODEL_PATH}")
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained(NLP_MODEL_PATH)
        mdl = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL_PATH)
        mdl.eval()
        return tok, mdl
    except Exception as e:
        st.warning(f"⚠️ NLP model failed to load: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# CHEXNET (chest X-ray — PyTorch DenseNet-121)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading CheXNet X-ray model…")
def get_chexnet():
    """
    Loads best_chexnet_multimodal.pth.
    Returns model or None if the .pth file is missing.
    """
    from model import load_vision_engine, VISION_WEIGHTS
    if not os.path.isfile(VISION_WEIGHTS):
        st.warning(f"⚠️ CheXNet weights not found at: {VISION_WEIGHTS}")
        return None
    try:
        return load_vision_engine()
    except Exception as e:
        st.warning(f"⚠️ CheXNet failed to load: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# KVASIR GI MODEL (EfficientNetB1 — TensorFlow/Keras)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading GI endoscopy model…")
def get_kvasir():
    """
    Loads gi_model_clean.h5 via TensorFlow/Keras.
    Returns model or None if TF is not installed or .h5 is missing.
    """
    from model import load_kvasir_engine, KVASIR_MODEL_PATH
    if not os.path.isfile(KVASIR_MODEL_PATH):
        st.warning(f"⚠️ Kvasir model not found at: {KVASIR_MODEL_PATH}")
        return None
    try:
        return load_kvasir_engine()
    except Exception as e:
        st.warning(f"⚠️ Kvasir model failed to load: {e}")
        return None