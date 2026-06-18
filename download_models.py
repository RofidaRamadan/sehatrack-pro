"""
download_models.py
==================
Called automatically at app startup to download model weights
from HuggingFace Hub into the container.

This solves the Railway/Streamlit Cloud memory problem:
- Git repos can't hold large .pth / .h5 files
- HuggingFace Hub stores them for free and serves them fast
- This script runs once; @st.cache_resource prevents re-downloading

HOW TO USE:
1. Run upload_models_to_hf.py ONCE on your local machine
2. Set these environment variables in your deployment platform:
      HF_TOKEN  = your HuggingFace read token
      HF_REPO   = your_username/sehatrack-weights
3. This script is called at the top of app.py automatically
"""

import os
import streamlit as st

HF_REPO  = os.environ.get("HF_REPO",  "Menna442004/sehatrack-weights")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

FILES_NEEDED = [
    "best_chexnet_multimodal.pth",
    "gi_model_clean.h5",
]
FOLDER_NEEDED = "model_only"   # NLP model directory


@st.cache_resource(show_spinner="📥 Downloading model weights (first run only)…")
def download_all_weights():
    """
    Downloads weights from HuggingFace Hub.
    @st.cache_resource ensures this runs ONLY ONCE per container lifetime.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        st.error("huggingface_hub not installed. Add it to requirements.txt")
        return False

    _HERE = os.path.dirname(os.path.abspath(__file__))
    all_ok = True

    # Download individual weight files
    for fname in FILES_NEEDED:
        dest = os.path.join(_HERE, fname)
        if os.path.isfile(dest):
            continue   # already present (e.g. mounted volume)
        try:
            print(f"[WeightLoader] Downloading {fname} …")
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=fname,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=_HERE,
            )
            print(f"[WeightLoader] ✅ {fname} → {path}")
        except Exception as e:
            st.warning(f"⚠️ Could not download {fname}: {e}")
            all_ok = False

    # Download the NLP model folder
    nlp_dest = os.path.join(_HERE, FOLDER_NEEDED)
    if not os.path.isdir(nlp_dest):
        try:
            print(f"[WeightLoader] Downloading NLP folder …")
            snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=_HERE,
                allow_patterns=f"{FOLDER_NEEDED}/*",
            )
            print(f"[WeightLoader] ✅ NLP folder downloaded")
        except Exception as e:
            st.warning(f"⚠️ Could not download NLP model folder: {e}")
            all_ok = False

    return all_ok