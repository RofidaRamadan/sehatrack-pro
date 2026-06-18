"""
model.py — SehaTrack Pro
Unified inference engine:
  ① NLP symptom classifier  (HuggingFace Transformers)
  ② CheXNet multimodal X-ray engine (DenseNet-121 + metadata branch)
  ③ Kvasir GI endoscopy engine  (EfficientNetB1 via TF/Keras)
  ④ Grad-CAM++ explainability for CheXNet
  ⑤ LIME explainability for Kvasir

All paths are relative — place your weight files next to this script:
    model_only/           ← HuggingFace NLP model directory
    best_chexnet_multimodal.pth
    gi_model_clean.h5
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import densenet121
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Lazy TF import so the app still loads if TF is not installed ──────────────
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    tf = None
    _TF_AVAILABLE = False

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# SHARED PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))

NLP_MODEL_PATH   = os.path.join(_HERE, "model_only")
VISION_WEIGHTS   = os.path.join(_HERE, "best_chexnet_multimodal.pth")
KVASIR_MODEL_PATH = os.path.join(_HERE, "gi_model_clean.h5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device

GI_CLASSES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

OPTIMAL_THRESHOLDS = {
    "Atelectasis": 0.38, "Cardiomegaly": 0.42, "Effusion": 0.40,
    "Infiltration": 0.35, "Mass": 0.30, "Nodule": 0.28,
    "Pneumonia": 0.45, "Pneumothorax": 0.33, "Consolidation": 0.37,
    "Edema": 0.41, "Emphysema": 0.29, "Fibrosis": 0.25,
    "Pleural_Thickening": 0.32, "Hernia": 0.20,
}


# ══════════════════════════════════════════════════════════════════════════════
# ① NLP SYMPTOM ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def _load_nlp():
    if not os.path.isdir(NLP_MODEL_PATH):
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained(NLP_MODEL_PATH)
        mdl = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL_PATH)
        mdl.to(device)
        mdl.eval()

        # Load id2label from JSON if not baked into config
        id2label_path = os.path.join(NLP_MODEL_PATH, "id2label.json")
        if os.path.isfile(id2label_path):
            with open(id2label_path) as f:
                extra = json.load(f)
            if not mdl.config.id2label:
                mdl.config.id2label = {int(k): v for k, v in extra.items()}

        return tok, mdl
    except Exception as e:
        print(f"[NLP] Failed to load: {e}")
        return None, None


_tokenizer, _nlp_model = _load_nlp()


def _get_probs(text: str) -> torch.Tensor:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _nlp_model(**inputs).logits
    return F.softmax(logits, dim=1)[0]


def predict(text: str) -> Tuple[str, float]:
    """Return (top_label, confidence) — backwards-compatible with original app.py."""
    if _nlp_model is None or not text.strip():
        return "Unknown Symptom", 0.0
    probs   = _get_probs(text)
    pred_id = torch.argmax(probs).item()
    label   = _nlp_model.config.id2label.get(pred_id, f"Class {pred_id}")
    return str(label), float(probs[pred_id])


def predict_topk(text: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return all symptoms sorted by confidence (desc).
    If k is given, return only top-k.
    Each item: {"rank": int, "label": str, "score": float}
    """
    if _nlp_model is None or not text.strip():
        return [{"rank": 1, "label": "Unknown Symptom", "score": 0.0}]

    probs      = _get_probs(text)
    sorted_ids = torch.argsort(probs, descending=True)
    if k is not None:
        sorted_ids = sorted_ids[:k]

    results = []
    for rank, idx in enumerate(sorted_ids, 1):
        score = float(probs[idx])
        if rank > 1 and score < 0.001:
            break
        results.append({
            "rank":  rank,
            "label": str(_nlp_model.config.id2label.get(idx.item(), f"Class {idx.item()}")),
            "score": score,
        })
    return results


def debug_nlp(text: str) -> None:
    """CLI helper: python -c "from model import debug_nlp; debug_nlp('I have a headache')" """
    if _nlp_model is None:
        print("NLP model not loaded.")
        return
    probs   = _get_probs(text)
    top_ids = torch.argsort(probs, descending=True)[:10]
    print(f"\n── Input: '{text}'")
    print(f"── Top 10 predictions ─────────────────")
    for rank, idx in enumerate(top_ids, 1):
        i     = idx.item()
        label = _nlp_model.config.id2label.get(i, f"[MISSING {i}]")
        print(f"  {rank:>2}. {label:<40s}  {probs[i]*100:6.2f}%")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ② CHEXNET MULTIMODAL X-RAY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def encode_meta(age: Any, gender: Any, view_pos: Any) -> torch.Tensor:
    """Encode patient metadata into a 3-value tensor for the CheXNet meta branch."""
    try:
        a = max(0.0, min(float(age), 120.0)) / 120.0
    except Exception:
        a = 0.0
    g = 1.0 if str(gender).lower().strip() == "female" else 0.0
    v = 1.0 if "ap" in str(view_pos).lower() else 0.0
    return torch.tensor([[a, g, v]], dtype=torch.float32)


class CheXNetMultimodal(nn.Module):
    def __init__(self, num_classes: int = 14, meta_dim: int = 3, dropout_rate: float = 0.4):
        super().__init__()
        base            = densenet121(weights=None)
        self.features   = base.features
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        dense_out       = base.classifier.in_features

        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
        )

        fusion = dense_out + 16
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fusion),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        feats = F.relu(self.features(img), inplace=False)
        x     = torch.flatten(self.avgpool(feats), 1)
        return self.classifier(torch.cat([x, self.meta_branch(meta)], dim=1))


def load_vision_engine(weights_path: str = VISION_WEIGHTS) -> CheXNetMultimodal:
    model = CheXNetMultimodal(num_classes=len(DISEASE_LABELS))
    if os.path.isfile(weights_path):
        try:
            ckpt  = torch.load(weights_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            if isinstance(state, dict):
                clean = {
                    k.replace("module.", "").replace("base_model.", ""): v
                    for k, v in state.items()
                }
                model.load_state_dict(clean, strict=False)
        except Exception as e:
            print(f"[CheXNet] Error loading weights: {e}")
    model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ③ GRAD-CAM++ EXPLAINABILITY FOR CHEXNET
# ══════════════════════════════════════════════════════════════════════════════
class GradCAMPlusPlus:
    def __init__(self, model: CheXNetMultimodal):
        self.model       = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Hook onto the final dense block for stable spatial resolution
        target = (
            model.features.denseblock4
            if hasattr(model.features, "denseblock4")
            else list(model.features.children())[-2]
        )
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _m, _i, output: torch.Tensor) -> None:
        self.activations = output.detach().clone()

    def _save_gradient(self, _m, _gi, grad_output: Tuple) -> None:
        self.gradients = grad_output[0].detach().clone()

    def generate(
        self,
        img_tensor:  torch.Tensor,
        meta_tensor: torch.Tensor,
        category_idx: int,
    ) -> Optional[np.ndarray]:
        self.model.zero_grad()
        img  = img_tensor.detach().clone().requires_grad_(True)
        meta = meta_tensor.detach().clone()

        with torch.enable_grad():
            output = self.model(img, meta)
            category_idx = int(category_idx)
            if not (0 <= category_idx < output.shape[1]):
                category_idx = int(torch.argmax(output, dim=1).item())
            output[0, category_idx].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        g2 = self.gradients ** 2
        g3 = self.gradients ** 3
        gsum = self.activations.sum(dim=(2, 3), keepdim=True)

        denom   = 2.0 * g2 + gsum * g3
        denom   = torch.where(denom != 0, denom, torch.ones_like(denom))
        alphas  = g2 / denom
        weights = (alphas * F.relu(self.gradients)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam).detach().cpu().numpy()

        max_val = cam.max()
        cam = cam / max_val if max_val > 1e-5 else np.zeros_like(cam)
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        cam = cv2.GaussianBlur(cam, (3, 3), 0)
        return cam

    # alias for callers that use generate_heatmap
    def generate_heatmap(self, img_tensor, meta_tensor, category_idx):
        return self.generate(img_tensor, meta_tensor, category_idx)

    @staticmethod
    def overlay(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
        base = np.array(pil_img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
        heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blend = (alpha * heat + (1.0 - alpha) * base).clip(0, 1)
        return Image.fromarray((blend * 255).astype(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
# ④ KVASIR GI ENDOSCOPY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def load_kvasir_engine():
    if not _TF_AVAILABLE:
        return None
    if os.path.isfile(KVASIR_MODEL_PATH):
        try:
            return tf.keras.models.load_model(KVASIR_MODEL_PATH)
        except Exception as e:
            print(f"[Kvasir] Failed to load saved model: {e}")

    # Fallback: build untrained architecture
    base = tf.keras.applications.EfficientNetB1(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    model = tf.keras.models.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(GI_CLASSES), activation="softmax"),
    ])
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ LIME EXPLAINABILITY FOR KVASIR
# ══════════════════════════════════════════════════════════════════════════════
def run_kvasir_lime_explanation(
    pil_image: Image.Image,
    trained_model,
    num_samples: int = 500,
) -> Tuple[str, float, List[Dict], Image.Image]:
    """
    Returns (class_name, confidence, chart_data, lime_boundary_image).
    Falls back gracefully if LIME is not installed.
    """
    img_array = np.array(pil_image.convert("RGB").resize((224, 224))).astype(np.float32)

    # Get base prediction
    batch      = tf.keras.applications.efficientnet.preprocess_input(
        np.expand_dims(img_array, 0)
    )
    preds      = trained_model.predict(batch, verbose=0)[0]
    top_idx    = int(np.argmax(preds))
    class_name = GI_CLASSES[top_idx]
    confidence = float(preds[top_idx])

    if not _LIME_AVAILABLE:
        return class_name, confidence, [], pil_image.resize((224, 224))

    def classifier_fn(images: np.ndarray) -> np.ndarray:
        proc = tf.keras.applications.efficientnet.preprocess_input(
            images.astype(np.float32)
        )
        return trained_model.predict(proc, verbose=0)

    explainer   = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array, classifier_fn,
        top_labels=1, hide_color=0, num_samples=num_samples,
    )

    dict_weights   = explanation.local_exp.get(top_idx, [])
    sorted_weights = sorted(dict_weights, key=lambda x: abs(x[1]), reverse=True)[:6]

    temp, mask = explanation.get_image_and_mask(
        top_idx, positive_only=True, num_features=6, hide_rest=False
    )
    boundary_img = Image.fromarray(
        (mark_boundaries(temp / 255.0, mask) * 255).astype(np.uint8)
    )

    chart_data = [
        {
            "Feature/Superpixel Segment": f"Segment Region ID #{seg_id}",
            "LIME Attribution Weight": float(w),
        }
        for seg_id, w in sorted_weights
    ]

    return class_name, confidence, chart_data, boundary_img























# # """
# # model.py — SehaTrack Pro
# # NLP symptom classifier + CheXNet multimodal X-ray model + Kvasir GI Engine
# # """

# # import os
# # from typing import Any, Dict, List, Optional, Tuple

# # import cv2
# # import numpy as np
# # import tensorflow as tf
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from PIL import Image
# # from torchvision.models import densenet121
# # from transformers import AutoModelForSequenceClassification, AutoTokenizer

# # from lime import lime_image
# # from skimage.segmentation import mark_boundaries

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # DEVICE = device

# # VISION_WEIGHTS = "best_chexnet_multimodal.pth"
# # KVASIR_MODEL_PATH = "gi_model_clean.h5"
# # NLP_MODEL_PATH = "model_only"

# # GI_CLASSES = [
# #     "dyed-lifted-polyps",
# #     "dyed-resection-margins",
# #     "esophagitis",
# #     "normal-cecum",
# #     "normal-pylorus",
# #     "normal-z-line",
# #     "polyps",
# #     "ulcerative-colitis",
# # ]

# # DISEASE_LABELS = [
# #     "Atelectasis",
# #     "Cardiomegaly",
# #     "Effusion",
# #     "Infiltration",
# #     "Mass",
# #     "Nodule",
# #     "Pneumonia",
# #     "Pneumothorax",
# #     "Consolidation",
# #     "Edema",
# #     "Emphysema",
# #     "Fibrosis",
# #     "Pleural_Thickening",
# #     "Hernia",
# # ]


# # def encode_meta(age: Any, gender: Any, view_pos: Any) -> torch.Tensor:
# #     """Encode age, gender, and view position into a 3-value tensor."""
# #     try:
# #         a = max(0.0, min(float(age), 120.0)) / 120.0
# #     except Exception:
# #         a = 0.0
# #     g = 1.0 if str(gender).lower().strip() == "female" else 0.0
# #     v = 1.0 if "ap" in str(view_pos).lower() else 0.0
# #     return torch.tensor([[a, g, v]], dtype=torch.float32)


# # try:
# #     tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL_PATH)
# #     nlp_model = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL_PATH)
# #     nlp_model.to(device)
# #     nlp_model.eval()
# # except Exception:
# #     tokenizer = None
# #     nlp_model = None


# # def predict(text: str) -> Tuple[str, float]:
# #     if tokenizer is None or nlp_model is None:
# #         return "Unknown Symptom", 0.0

# #     cleaned = str(text).strip()
# #     if not cleaned:
# #         return "Unknown Symptom", 0.0

# #     inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=512).to(device)
# #     with torch.no_grad():
# #         outputs = nlp_model(**inputs)
# #         probs = F.softmax(outputs.logits, dim=1)
# #         pred_idx = int(torch.argmax(probs, dim=1).item())
# #         conf = float(probs[0, pred_idx].item())

# #     label_map = getattr(nlp_model.config, "id2label", None)
# #     if isinstance(label_map, dict) and pred_idx in label_map:
# #         predicted_label = str(label_map[pred_idx])
# #     elif isinstance(label_map, dict) and str(pred_idx) in label_map:
# #         predicted_label = str(label_map[str(pred_idx)])
# #     else:
# #         predicted_label = f"Symptom Class {pred_idx}"

# #     return predicted_label, conf


# # class CheXNetMultimodal(nn.Module):
# #     def __init__(self, num_classes: int = 14, meta_dim: int = 3, dropout_rate: float = 0.4):
# #         super().__init__()
# #         base = densenet121(weights=None)
# #         self.features = base.features
# #         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# #         dense_out = base.classifier.in_features

# #         self.meta_branch = nn.Sequential(
# #             nn.Linear(meta_dim, 32),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(0.2),
# #             nn.Linear(32, 16),
# #             nn.ReLU(inplace=True),
# #         )

# #         fusion = dense_out + 16
# #         self.classifier = nn.Sequential(
# #             nn.BatchNorm1d(fusion),
# #             nn.Dropout(dropout_rate),
# #             nn.Linear(fusion, 512),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(dropout_rate / 2),
# #             nn.Linear(512, num_classes),
# #         )

# #     def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
# #         feats = self.features(img)
# #         feats = F.relu(feats, inplace=False)
# #         x = self.avgpool(feats)
# #         x = torch.flatten(x, 1)
# #         meta_out = self.meta_branch(meta)
# #         return self.classifier(torch.cat([x, meta_out], dim=1))


# # class GradCAMPlusPlus:
# #     def __init__(self, model: nn.Module):
# #         self.model = model
# #         self.gradients: Optional[torch.Tensor] = None
# #         self.activations: Optional[torch.Tensor] = None

# #         # DenseNet norm5 output is a stable feature map for CAM.
# #         target_layer = self.model.features.norm5
# #         target_layer.register_forward_hook(self.save_activation)
# #         target_layer.register_full_backward_hook(self.save_gradient)

# #     def save_activation(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
# #         self.activations = output.detach()

# #     def save_gradient(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
# #         self.gradients = grad_output[0].detach()

# #     def generate(self, img_tensor: torch.Tensor, meta_tensor: torch.Tensor, category_idx: int) -> Optional[np.ndarray]:
# #         if self.model is None:
# #             return None

# #         self.model.zero_grad(set_to_none=True)

# #         image = img_tensor.detach().clone()
# #         image.requires_grad_(True)
# #         meta = meta_tensor.detach().clone()

# #         with torch.enable_grad():
# #             output = self.model(image, meta)
# #             category_idx = int(category_idx)
# #             if category_idx < 0 or category_idx >= output.shape[1]:
# #                 category_idx = int(torch.argmax(output, dim=1).item())
# #             logit = output[0, category_idx]
# #             logit.backward(retain_graph=True)

# #         gradients = self.gradients
# #         activations = self.activations
# #         if gradients is None or activations is None:
# #             return None

# #         gradients_2 = gradients.pow(2)
# #         gradients_3 = gradients.pow(3)
# #         global_sum = activations.sum(dim=(2, 3), keepdim=True)

# #         alpha_denom = (2.0 * gradients_2) + (global_sum * gradients_3)
# #         alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
# #         alphas = gradients_2 / alpha_denom

# #         weights = (alphas * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
# #         cam = (weights * activations).sum(dim=1).squeeze(0)
# #         cam = torch.relu(cam).detach().cpu().numpy()

# #         if cam.max() > 0:
# #             cam = cam / cam.max()

# #         cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
# #         cam = cv2.GaussianBlur(cam, (3, 3), 0)
# #         return cam

# #     def generate_heatmap(self, img_tensor: torch.Tensor, meta_tensor: torch.Tensor, category_idx: int) -> Optional[np.ndarray]:
# #         return self.generate(img_tensor, meta_tensor, category_idx)

# #     @staticmethod
# #     def overlay(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
# #         base = np.array(pil_img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
# #         heat = np.clip(cam, 0.0, 1.0)
# #         heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
# #         heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
# #         blend = (alpha * heat + (1.0 - alpha) * base).clip(0, 1)
# #         return Image.fromarray((blend * 255).astype(np.uint8))


# # def load_vision_engine(weights_path: str = VISION_WEIGHTS) -> CheXNetMultimodal:
# #     model = CheXNetMultimodal(num_classes=len(DISEASE_LABELS))
# #     if os.path.exists(weights_path):
# #         try:
# #             ckpt = torch.load(weights_path, map_location=device)
# #             state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
# #             if isinstance(ckpt, dict) and state is None:
# #                 state = ckpt.get("state_dict", ckpt)
# #             clean_state = {}
# #             if isinstance(state, dict):
# #                 for k, v in state.items():
# #                     new_key = k.replace("module.", "").replace("base_model.", "")
# #                     clean_state[new_key] = v
# #                 model.load_state_dict(clean_state, strict=False)
# #         except Exception as e:
# #             print(f"Error loading vision parameters: {e}")
# #     model.to(device)
# #     model.eval()
# #     return model


# # def load_kvasir_engine() -> tf.keras.Model:
# #     if os.path.exists(KVASIR_MODEL_PATH):
# #         return tf.keras.models.load_model(KVASIR_MODEL_PATH)

# #     img_size = (224, 224, 3)
# #     base_model = tf.keras.applications.EfficientNetB1(input_shape=img_size, include_top=False, weights=None)
# #     model = tf.keras.models.Sequential(
# #         [
# #             base_model,
# #             tf.keras.layers.GlobalAveragePooling2D(),
# #             tf.keras.layers.BatchNormalization(),
# #             tf.keras.layers.Dense(256, activation="relu"),
# #             tf.keras.layers.Dropout(0.4),
# #             tf.keras.layers.Dense(len(GI_CLASSES), activation="softmax"),
# #         ]
# #     )
# #     return model


# # def run_kvasir_lime_explanation(pil_image: Image.Image, trained_model: tf.keras.Model):
# #     img_array = np.array(pil_image.convert("RGB").resize((224, 224))).astype(np.float32)

# #     def classifier_fn(images: np.ndarray) -> np.ndarray:
# #         images_np = np.asarray(images, dtype=np.float32)
# #         preprocessed = tf.keras.applications.efficientnet.preprocess_input(images_np)
# #         return trained_model.predict(preprocessed, verbose=0)

# #     explainer = lime_image.LimeImageExplainer()
# #     explanation = explainer.explain_instance(
# #         img_array,
# #         classifier_fn,
# #         top_labels=1,
# #         hide_color=0,
# #         num_samples=1000,
# #     )

# #     batch_img = np.expand_dims(img_array, axis=0)
# #     preprocessed_batch = tf.keras.applications.efficientnet.preprocess_input(batch_img)
# #     preds = trained_model.predict(preprocessed_batch, verbose=0)[0]
# #     top_label_idx = int(np.argmax(preds))

# #     dict_weights = explanation.local_exp.get(top_label_idx, [])
# #     sorted_weights = sorted(dict_weights, key=lambda x: abs(x[1]), reverse=True)[:6]

# #     temp, mask = explanation.get_image_and_mask(top_label_idx, positive_only=True, num_features=6, hide_rest=False)
# #     img_boundaries = mark_boundaries(temp / 255.0, mask)
# #     img_boundaries = Image.fromarray((img_boundaries * 255).astype(np.uint8))

# #     chart_data = []
# #     for seg_id, weight in sorted_weights:
# #         chart_data.append(
# #             {
# #                 "Feature/Superpixel Segment": f"Segment Region ID #{seg_id}",
# #                 "LIME Attribution Weight": float(weight),
# #             }
# #         )

# #     return GI_CLASSES[top_label_idx], float(preds[top_label_idx]), chart_data, img_boundaries










# """
# model.py — SehaTrack Pro
# NLP symptom classifier + CheXNet multimodal X-ray model + Kvasir GI Engine
# """

# import os
# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from PIL import Image
# from torchvision.models import densenet121
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# import tensorflow as tf
# from lime import lime_image
# from skimage.segmentation import mark_boundaries

# # ══════════════════════════════════════════════════════════════════════════════
# # SHARED ENVIRONMENT CONTEXT & WEIGHT PATHS
# # ══════════════════════════════════════════════════════════════════════════════
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = device

# VISION_WEIGHTS = "best_chexnet_multimodal.pth"
# KVASIR_MODEL_PATH = "gi_model_clean.h5"
# NLP_MODEL_PATH = r"C:\Users\RofaR\OneDrive\Desktop\ManarGP\model_only"

# # Kvasir GI Labels
# GI_CLASSES = [
#     'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 
#     'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis'
# ]

# # CheXNet Labels
# DISEASE_LABELS = [
#     'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
#     'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
#     'Fibrosis', 'Pleural_Thickening', 'Hernia'
# ]

# OPTIMAL_THRESHOLDS = {
#     "Atelectasis": 0.38, "Cardiomegaly": 0.42, "Effusion": 0.40, "Infiltration": 0.35,
#     "Mass": 0.30, "Nodule": 0.28, "Pneumonia": 0.45, "Pneumothorax": 0.33,
#     "Consolidation": 0.37, "Edema": 0.41, "Emphysema": 0.29, "Fibrosis": 0.25,
#     "Pleural_Thickening": 0.32, "Hernia": 0.20
# }

# def encode_meta(age, gender, view_pos):
#     """Encodes patient metadata to map perfectly into the multimodal CheXNet branch."""
#     a = min(float(age), 120.0) / 120.0
#     g = 1.0 if str(gender).lower() == 'female' else 0.0
#     v = 1.0 if 'AP' in str(view_pos) else 0.0
#     return torch.tensor([[a, g, v]], dtype=torch.float32, device=device)

# # ══════════════════════════════════════════════════════════════════════════════
# # ① NLP SYMPTOM ENGINE
# # ══════════════════════════════════════════════════════════════════════════════
# try:
#     if os.path.exists(NLP_MODEL_PATH):
#         tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL_PATH)
#         nlp_model = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL_PATH)
#         nlp_model.to(device)
#         nlp_model.eval()
#     else:
#         tokenizer, nlp_model = None, None
# except Exception:
#     tokenizer, nlp_model = None, None

# def predict(text):
#     if not nlp_model or not text.strip():
#         return "Unknown Symptom", 0.90
        
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
#     with torch.no_grad():
#         outputs = nlp_model(**inputs)
#         probs = F.softmax(outputs.logits, dim=1)
#         pred_idx = torch.argmax(probs, dim=1).item()
#         conf = probs[0][pred_idx].item()
    
#     if nlp_model.config.id2label:
#         predicted_label = nlp_model.config.id2label.get(pred_idx, f"Class {pred_idx}")
#     else:
#         predicted_label = f"Symptom Class {pred_idx}"
        
#     return predicted_label, conf

# # ══════════════════════════════════════════════════════════════════════════════
# # ② CHEXNET CHEST X-RAY MULTI-LABEL ENGINE & GRAD-CAM++
# # ══════════════════════════════════════════════════════════════════════════════
# class CheXNetMultimodal(nn.Module):
#     def __init__(self, num_classes=14, meta_dim=3, dropout_rate=0.4):
#         super().__init__()
#         base = densenet121(pretrained=False)
#         self.features = base.features
#         self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
#         dense_out     = base.classifier.in_features
        
#         self.meta_branch = nn.Sequential(
#             nn.Linear(meta_dim, 32), nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 16), nn.ReLU(),
#         )
        
#         fusion = dense_out + 16
#         self.classifier = nn.Sequential(
#             nn.BatchNorm1d(fusion),
#             nn.Dropout(dropout_rate),
#             nn.Linear(fusion, 512), nn.ReLU(),
#             nn.Dropout(dropout_rate / 2),
#             nn.Linear(512, num_classes)
#         )
        
#     def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
#         feats = self.features(img)
#         out = F.relu(feats, inplace=False) 
#         x = self.avgpool(out)
#         x = torch.flatten(x, 1)
#         meta_out = self.meta_branch(meta)
#         return torch.sigmoid(self.classifier(torch.cat([x, meta_out], dim=1)))

# class GradCAMPlusPlus:
#     def __init__(self, model):
#         self.model = model
#         self.gradients = None
#         self.activations = None
        
#         # FIX: Target final convolutional features (denseblock4) to preserve accurate spatial resolution
#         if hasattr(self.model.features, 'denseblock4'):
#             target_layer = self.model.features.denseblock4
#         else:
#             target_layer = list(self.model.features.children())[-2]
            
#         target_layer.register_forward_hook(self.save_activation)
#         target_layer.register_full_backward_hook(self.save_gradient)

#     def save_activation(self, module, input, output):
#         self.activations = output.clone().detach()

#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].clone().detach()

#     def generate(self, img_tensor, meta_tensor, category_idx):
#         self.model.zero_grad()
#         if not img_tensor.requires_grad:
#             img_tensor.requires_grad = True
            
#         output = self.model(img_tensor, meta_tensor)
#         logit = output[0, category_idx]
#         logit.backward()
        
#         gradients = self.gradients
#         activations = self.activations
        
#         gradients_2 = gradients ** 2
#         gradients_3 = gradients ** 3
#         global_sum = activations.sum(dim=[2, 3], keepdim=True)
        
#         alpha_num = gradients_2
#         alpha_denom = (2.0 * gradients_2) + (global_sum * gradients_3)
#         alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
#         alphas = alpha_num / alpha_denom
        
#         weights = (alphas * F.relu(gradients)).sum(dim=[2, 3], keepdim=True)
#         weighted_activations = activations * weights
        
#         heatmap = weighted_activations.sum(dim=1).squeeze().cpu().numpy()
#         heatmap = np.maximum(heatmap, 0)
        
#         # FIX: Noise filter cutoff prevents tiny ambient fluctuations from stretching into an entire blue/green wash
#         max_val = np.max(heatmap)
#         if max_val > 1e-5:
#             heatmap /= max_val
#         else:
#             heatmap = np.zeros_like(heatmap)
            
#         return cv2.resize(heatmap, (224, 224))

#     def generate_heatmap(self, img_tensor, meta_tensor, category_idx):
#         """Interface safety fallback to eliminate potential missing attribute method errors."""
#         return self.generate(img_tensor, meta_tensor, category_idx)

#     @staticmethod
#     def overlay(pil_img, cam, alpha=0.45):
#         base = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
#         if len(base.shape) == 2:
#             base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
#         elif base.shape[2] == 4:
#             base = cv2.cvtColor(base, cv2.COLOR_RGBA2RGB)
            
#         heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
#         heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
#         blend = (alpha * heat + (1 - alpha) * base).clip(0, 1)
#         return Image.fromarray((blend * 255).astype(np.uint8))

# def load_vision_engine(weights_path=VISION_WEIGHTS):
#     model = CheXNetMultimodal(num_classes=14)
#     if os.path.exists(weights_path):
#         try:
#             ckpt = torch.load(weights_path, map_location=device, weights_only=False)
#             state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
#             clean_state = {k.replace("module.", "").replace("base_model.", ""): v for k, v in state.items()}
#             model.load_state_dict(clean_state, strict=False)
#         except Exception as e:
#             print(f"Error loading vision parameters: {e}")
#     model.to(device)
#     model.eval()
#     return model

# # ══════════════════════════════════════════════════════════════════════════════
# # ③ GASTROSCOPY (KVASIR) ENGINE & LIME EXPLAINER
# # ══════════════════════════════════════════════════════════════════════════════
# def load_kvasir_engine():
#     if os.path.exists(KVASIR_MODEL_PATH):
#         try:
#             return tf.keras.models.load_model(KVASIR_MODEL_PATH)
#         except Exception:
#             pass
            
#     img_size = (224, 224, 3)
#     base_model = tf.keras.applications.EfficientNetB1(input_shape=img_size, include_top=False, weights=None)
#     m = tf.keras.models.Sequential([
#         base_model,
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dropout(0.4),
#         tf.keras.layers.Dense(8, activation='softmax')
#     ])
#     return m

# def run_kvasir_lime_explanation(pil_image, trained_model):
#     """Executes full LIME context region calculations on the target gastrointestinal image input."""
#     img_array = np.array(pil_image.convert("RGB").resize((224, 224))).astype(np.float32)

#     def classifier_fn(images):
#         preprocessed = tf.keras.applications.efficientnet.preprocess_input(images.astype(np.float32))
#         return trained_model.predict(preprocessed, verbose=0)
        
#     explainer = lime_image.LimeImageExplainer()
#     explanation = explainer.explain_instance(img_array, classifier_fn, top_labels=1, hide_color=0, num_samples=50)
    
#     batch_img = np.expand_dims(img_array, axis=0)
#     preprocessed_batch = tf.keras.applications.efficientnet.preprocess_input(batch_img)
#     preds = trained_model.predict(preprocessed_batch, verbose=0)[0]
#     top_label_idx = np.argmax(preds)
    
#     dict_weights = explanation.local_exp[top_label_idx]
#     sorted_weights = sorted(dict_weights, key=lambda x: abs(x[1]), reverse=True)[:6]
    
#     temp, mask = explanation.get_image_and_mask(top_label_idx, positive_only=True, num_features=3, hide_rest=False)
#     img_boundaries = mark_boundaries(temp / 255.0, mask)
#     img_boundaries = Image.fromarray((img_boundaries * 255).astype(np.uint8))
    
#     chart_data = []
#     for seg_id, weight in sorted_weights:
#         chart_data.append({
#             "Feature/Superpixel Segment": f"Segment Region ID #{seg_id}",
#             "LIME Attribution Weight": float(weight)
#         })
#     return GI_CLASSES[top_label_idx], float(preds[top_label_idx]), chart_data, img_boundaries