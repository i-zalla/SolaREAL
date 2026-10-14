import time
from pathlib import Path
import tempfile
import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import librosa
import joblib
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, ViTImageProcessor, ViTModel
from ultralytics import YOLO

import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Detector — SolaREAL",
    page_icon="✦",
    layout="wide",
)

# ============================================================
# CONFIG - Keep in sync with training
# ============================================================

class Config:
    features_dim = 256
    num_classes = 2
    num_stages = 2
    num_layers = 4
    num_f_maps = 64
    kernel_size = 1
    dropout = 0.7

    xlsr_model_name = "facebook/wav2vec2-xls-r-300m"
    vit_model_name = 'google/vit-base-patch16-224-in21k'

    attention_fusion_seed = 42

    vit_batch_size = 16
    xlsr_batch_size = 32
    use_fp16 = True  


# ============================================================
# PLATFORM MODEL PATHS - ROBUST VERSION
# ============================================================

script_dir = Path(__file__).parent  
project_root = script_dir.parent     

models_dir_1 = project_root / "models"

models_dir_2 = script_dir / "models"

models_dir_3 = Path.cwd() / "models"

if models_dir_1.exists():
    MODELS_ROOT = models_dir_1
    print(f"Using models from: {MODELS_ROOT}")
elif models_dir_2.exists():
    MODELS_ROOT = models_dir_2
    print(f"Using models from: {MODELS_ROOT}")
elif models_dir_3.exists():
    MODELS_ROOT = models_dir_3
    print(f"Using models from: {MODELS_ROOT}")
else:
    MODELS_ROOT = models_dir_1  
    print(f"Warning: No models directory found. Looking in: {MODELS_ROOT}")

YOLO_WEIGHTS = MODELS_ROOT / "yolov8m-face-lindevs.pt"
SCALER_PATH = MODELS_ROOT / "scaler.save"
MODEL_PATH = MODELS_ROOT / "best_model.pth"
ATTENTION_FUSION_PATH = MODELS_ROOT / "attention_fusion.pth"

print(f"Script location: {script_dir}")
print(f"Project root: {project_root}")
print(f"Models root: {MODELS_ROOT}")
print(f"YOLO weights path: {YOLO_WEIGHTS}")
print(f"YOLO weights exists: {YOLO_WEIGHTS.exists()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = Config.use_fp16 and device.type == "cuda"


# ============================================================
# MODEL ARCHITECTURE 
# ============================================================

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Dropout(dropout)
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class AV_MSTCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(config['features_dim'], config['num_f_maps'], 1),
            nn.BatchNorm1d(config['num_f_maps']),
            nn.GELU()
        )
        self.stages = nn.ModuleList([
            nn.Sequential(*[
                DilatedResidualLayer(2 ** i, config['num_f_maps'], config['num_f_maps'],
                                      config['kernel_size'], config['dropout'])
                for i in range(config['num_layers'])
            ]) for _ in range(config['num_stages'])
        ])
        self.attention = nn.Sequential(
            nn.Conv1d(config['num_f_maps'], config['num_f_maps'] // 4, 1),
            nn.GELU(),
            nn.Conv1d(config['num_f_maps'] // 4, 1, 1),
            nn.Softmax(dim=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(config['num_f_maps'], config['num_f_maps'] // 2),
            nn.LayerNorm(config['num_f_maps'] // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config['num_f_maps'] // 2, config['num_classes'])
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        for stage in self.stages:
            x = stage(x)
        attn_weights = self.attention(x)
        x = torch.sum(x * attn_weights, dim=2)
        return self.classifier(x)


class AttentionFusion(nn.Module):
    def __init__(self, visual_dim=768, audio_dim=1024, hidden_dim=256):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim * 2, 2)
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, visual_feat, audio_feat):
        v = self.visual_proj(visual_feat)
        a = self.audio_proj(audio_feat)
        combined = torch.cat([v, a], dim=1)
        attention_weights = F.softmax(self.attention(combined), dim=1)
        attended_visual = attention_weights[:, 0:1] * v
        attended_audio = attention_weights[:, 1:2] * a
        fused = torch.cat([attended_visual, attended_audio], dim=1)
        return self.fusion_proj(fused)


# ============================================================
# MODEL LOADING FUNCTIONS
# ============================================================

@st.cache_resource
def get_attention_fusion():
    """Load or create AttentionFusion with persistence."""
    if ATTENTION_FUSION_PATH.exists():
        fusion = AttentionFusion().to(device)
        fusion.load_state_dict(torch.load(ATTENTION_FUSION_PATH, map_location=device, weights_only=False))
        print(f"Loaded existing AttentionFusion weights from {ATTENTION_FUSION_PATH}")
    else:
        torch.manual_seed(Config.attention_fusion_seed)
        random.seed(Config.attention_fusion_seed)
        np.random.seed(Config.attention_fusion_seed)
        fusion = AttentionFusion().to(device)
        ATTENTION_FUSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fusion.state_dict(), ATTENTION_FUSION_PATH)
        print(f"Created new AttentionFusion weights (seed={Config.attention_fusion_seed}) and saved to {ATTENTION_FUSION_PATH}")
    fusion.eval()
    return fusion


@st.cache_resource
def get_client_models():
    """Load all required models."""

    if not MODELS_ROOT.exists():
        raise FileNotFoundError(
            f"Models directory not found: {MODELS_ROOT}\n"
            f"Please create a 'models' folder at one of these locations:\n"
            f"  - {models_dir_1}\n"
            f"  - {models_dir_2}\n"
            f"  - {models_dir_3}\n"
            f"And place your model files there."
        )

    print("Loading ViT...")
    vit_processor = ViTImageProcessor.from_pretrained(Config.vit_model_name)
    vit_model = ViTModel.from_pretrained(Config.vit_model_name).to(device).eval()

    print("Loading YOLO face model...")
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {YOLO_WEIGHTS}\n"
            f"Please ensure the file exists in one of these locations:\n"
            f"  - {models_dir_1 / 'yolov8m-face-lindevs.pt'}\n"
            f"  - {models_dir_2 / 'yolov8m-face-lindevs.pt'}\n"
            f"  - {models_dir_3 / 'yolov8m-face-lindevs.pt'}"
        )

    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load

    try:
        yolo_face_model = YOLO(str(YOLO_WEIGHTS))
        yolo_face_model.to(device) 
    finally:
        torch.load = original_torch_load

    print("Loading XLS-R...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(Config.xlsr_model_name)
    xlsr_model = Wav2Vec2Model.from_pretrained(Config.xlsr_model_name).to(device).eval()

    print("Loading AttentionFusion...")
    attention_fusion = get_attention_fusion()

    print("Loading classifier...")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Classifier model not found: {MODEL_PATH}")

    model = AV_MSTCN({
        'features_dim': Config.features_dim,
        'num_classes': Config.num_classes,
        'num_stages': Config.num_stages,
        'num_layers': Config.num_layers,
        'num_f_maps': Config.num_f_maps,
        'kernel_size': Config.kernel_size,
        'dropout': Config.dropout
    }).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return (
        vit_processor,
        vit_model,
        yolo_face_model,
        feature_extractor,
        xlsr_model,
        attention_fusion,
        scaler,
        model,
    )


def preprocess_image_for_vit(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


def extract_vit_features_batch(images, vit_processor, vit_model, batch_size=Config.vit_batch_size):
    """
    images: list of preprocessed (224x224x3 RGB) numpy arrays.
    Returns: np.ndarray of shape (len(images), 768), or an empty array if `images` is empty.

    Runs ViT over chunks of `batch_size` images per forward pass instead of
    one image per call — this is where most of the GPU-utilization win comes
    from, since a single-image forward pass barely uses the 4060's compute
    before control returns to Python.
    """
    if not images:
        return np.zeros((0, 768), dtype=np.float32)

    all_feats = []
    try:
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            inputs = vit_processor(images=chunk, return_tensors="pt").to(device)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                outputs = vit_model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
            all_feats.append(feats)
        return np.concatenate(all_feats, axis=0)
    except Exception as e:
        print(f"Error in batched ViT feature extraction: {e}")
        return np.zeros((0, 768), dtype=np.float32)


def extract_xlsr_features(audio_path, feature_extractor, xlsr_model, fps=24, batch_size=Config.xlsr_batch_size):
    """
    Same output shape/semantics as the original per-frame version
    ((total_frames, 1024) array, one XLS-R embedding per video frame slot),
    but chunks are padded and batched through the model instead of being
    fed one at a time.
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        samples_per_frame = int(sr * (1.0 / fps))
        total_frames = len(y) // samples_per_frame
        if total_frames == 0:
            return np.array([])

        features = np.zeros((total_frames, 1024), dtype=np.float32)

        chunks = []
        chunk_indices = []
        for i in range(total_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            chunk = y[start:end]
            if len(chunk) == 0:
                continue
            chunks.append(chunk)
            chunk_indices.append(i)

        for b_start in range(0, len(chunks), batch_size):
            b_chunks = chunks[b_start:b_start + batch_size]
            b_indices = chunk_indices[b_start:b_start + batch_size]
            inputs = feature_extractor(
                b_chunks, sampling_rate=sr, return_tensors="pt", padding=True
            ).to(device)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                outputs = xlsr_model(**inputs)
            # mean-pool over time per sample in the batch
            pooled = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
            for local_idx, frame_idx in enumerate(b_indices):
                features[frame_idx] = pooled[local_idx]

        return features
    except Exception as e:
        print(f"Error in audio processing: {e}")
        return np.array([])


def find_face_crop_with_face_mesh(frame, face_mesh):
    """CPU-bound landmark detection — returns a preprocessed 224x224 RGB
    crop, or None. No GPU work happens here; ViT runs later, in a batch."""
    try:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame.shape
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            nose_bottom_y = face_landmarks.landmark[4].y * img_h
            chin_y = face_landmarks.landmark[152].y * img_h
            crop_height = chin_y - nose_bottom_y
            top_crop = int(nose_bottom_y)
            bottom_crop = int(chin_y - 0.05 * crop_height)
            x_coords = [lm.x * img_w for lm in face_landmarks.landmark if lm.y * img_h >= nose_bottom_y]
            left_crop = int(max(0, min(x_coords))) if x_coords else 0
            right_crop = int(min(img_w, max(x_coords))) if x_coords else img_w
            pad = int(0.05 * (right_crop - left_crop))
            left_crop = max(0, left_crop - pad)
            right_crop = min(img_w, right_crop + pad)
            if top_crop < bottom_crop and right_crop > left_crop:
                cropped = frame[top_crop:bottom_crop, left_crop:right_crop]
                if cropped.size > 0:
                    return preprocess_image_for_vit(cropped)
        return None
    except Exception as e:
        print(f"Error in Face Mesh processing: {e}")
        return None


def find_face_crop_with_yolo(frame, yolo_face_model):
    """Fallback face localization via YOLO. Also CPU/GPU-detection-only —
    no ViT call here either, so it can share the same batching path."""
    try:
        results = yolo_face_model.predict(frame, conf=0.25, verbose=False, device=device)
        if len(results) == 0:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        largest_box = max(xyxy, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = largest_box.astype(int)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        return preprocess_image_for_vit(face_crop)
    except Exception as e:
        print(f"YOLO fallback error: {e}")
        return None


def process_video(video_path, face_mesh, yolo_face_model, vit_processor, vit_model,
                   feature_extractor, xlsr_model, attention_fusion):
    audio_path = "temp_audio_single.wav"
    os.system(f"ffmpeg -y -i \"{video_path}\" -ar 16000 -ac 1 -loglevel error {audio_path}")
    audio_features = extract_xlsr_features(audio_path, feature_extractor, xlsr_model)
    if os.path.exists(audio_path):
        os.remove(audio_path)

    if len(audio_features) == 0:
        raise RuntimeError(f"No audio features extracted for {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24
    frame_interval = max(1, int(fps / 24))

    crops = []
    crop_frame_indices = []
    frame_count = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_index = processed_frames
            if frame_index >= len(audio_features):
                break
            crop = find_face_crop_with_face_mesh(frame, face_mesh)
            if crop is None:
                crop = find_face_crop_with_yolo(frame, yolo_face_model)
            if crop is not None:
                crops.append(crop)
                crop_frame_indices.append(frame_index)
            processed_frames += 1
        frame_count += 1

    cap.release()

    if not crops:
        raise RuntimeError(f"No frames with a detected face in {video_path}")

    visual_features = extract_vit_features_batch(crops, vit_processor, vit_model)

    matched_audio = audio_features[crop_frame_indices]  # (N, 1024)
    visual_tensor = torch.from_numpy(visual_features).float().to(device)
    audio_tensor = torch.from_numpy(matched_audio).float().to(device)

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
        fused = attention_fusion(visual_tensor, audio_tensor)
    fused = fused.float().cpu().numpy()  # (N, 256)

    if device.type == "cuda":
        torch.cuda.empty_cache() 

    return fused


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_video(video_path):
    (
        vit_processor,
        vit_model,
        yolo_face_model,
        feature_extractor,
        xlsr_model,
        attention_fusion,
        scaler,
        model,
    ) = get_client_models()

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as face_mesh:

        features = process_video(
            video_path,
            face_mesh,
            yolo_face_model,
            vit_processor,
            vit_model,
            feature_extractor,
            xlsr_model,
            attention_fusion,
        )

    features_scaled = scaler.transform(features)

    x = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
        outputs = model(x)
        probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    pred_label = "fake" if pred == 0 else "real"

    return {
        "label": pred_label,
        "fake_probability": float(probs[0]),
        "real_probability": float(probs[1]),
        "frames_analyzed": int(features.shape[0]),
        "device": str(device),
    }


# ============================================================
# AUTH GATE — SESSION STATE
# ------------------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "pending_video_path" not in st.session_state:
    st.session_state.pending_video_path = None
if "pending_video_name" not in st.session_state:
    st.session_state.pending_video_name = None
if "auto_run_analysis" not in st.session_state:
    st.session_state.auto_run_analysis = False

LOGIN_PAGE = "pages/Login.py"


def run_and_display_analysis(video_path):
    """Runs the deepfake analysis on video_path and renders the result UI."""
    start = time.perf_counter()

    try:
        with st.status(
            "Running deepfake analysis...",
            expanded=True,
        ) as status:
            st.write("Loading detection models...")
            result = analyze_video(video_path)
            elapsed = time.perf_counter() - start
            status.update(
                label="Analysis complete",
                state="complete",
                expanded=False,
            )

        fake = result["fake_probability"] * 100
        real = result["real_probability"] * 100
        label = result["label"].upper()

        st.markdown(
            '<div class="result-section">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="result-title">Analysis result</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-ring">{real:.1f}%</div>
                    <div class="score-label">REAL</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-ring">{fake:.1f}%</div>
                    <div class="score-label">FAKE</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div class="verdict-card">
                    <div class="verdict-label">VERDICT</div>
                    <div class="verdict">{label}</div>
                    <div class="verdict-sub">
                        {result["frames_analyzed"]} analyzed frame features
                        · {elapsed:.1f}s
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Technical details"):
            st.json(result)

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)

    finally:
        if Path(video_path).exists():
            Path(video_path).unlink()


# ============================================================
# UI — SolaREAL DESIGN
# ============================================================

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="brand"><span class="brand-mark">✦</span><span class="brand-name">SolaREAL</span></div>
  <div class="nav-links">
    <a href="/?page=detector">Detector</a>
    <a href="/?page=pricing">Pricing</a>
    <a href="/?page=technology">Technology</a>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<h1 class="page-title">Deepfake detector</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="page-subtitle">Upload a video and analyze its audio-visual content.</p>',
    unsafe_allow_html=True,
)

if st.session_state.logged_in:
    acc_col1, acc_col2 = st.columns([5, 1])
    with acc_col1:
        st.caption(f"Signed in as **{st.session_state.username}**")
    with acc_col2:
        if st.button("Log out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

uploaded = st.file_uploader(
    "Upload video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
)

if uploaded:
    st.video(uploaded)

    if st.button(
        "ANALYZE VIDEO",
        type="primary",
        use_container_width=True,
    ):
        suffix = Path(uploaded.name).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:
            tmp.write(uploaded.getbuffer())
            saved_path = tmp.name

        if not st.session_state.logged_in:
            st.session_state.pending_video_path = saved_path
            st.session_state.pending_video_name = uploaded.name
            st.session_state.auto_run_analysis = True
            st.switch_page(LOGIN_PAGE)
        else:
            run_and_display_analysis(saved_path)

elif st.session_state.logged_in and st.session_state.auto_run_analysis and st.session_state.pending_video_path:
    st.info(
        f"Welcome back! Resuming analysis of "
        f"**{st.session_state.pending_video_name}**…"
    )
    st.session_state.auto_run_analysis = False
    video_path = st.session_state.pending_video_path
    st.session_state.pending_video_path = None
    st.session_state.pending_video_name = None
    run_and_display_analysis(video_path)
