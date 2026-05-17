import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import pandas as pd

st.set_page_config(
    page_title="PPE Safety · Vision",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background: #faf7f2;
}

section[data-testid="stSidebar"] {
    background: #faf7f2;
    border-right: 1px solid #e2dbd0;
}
section[data-testid="stSidebar"] * {
    color: #3d3530 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #3d3530 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #e2dbd0 !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label {
    font-size: 14px !important;
    color: #5c5049 !important;
}

/* Page header */
.page-header {
    padding-bottom: 1.6rem;
    border-bottom: 2px solid #3d3530;
    margin-bottom: 2.4rem;
}
.page-eyebrow {
    font-family: 'Lato', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9c7c5a;
    margin-bottom: 6px;
}
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 30px;
    font-weight: 500;
    color: #3d3530;
    line-height: 1.15;
    margin: 0;
}
.page-desc {
    font-size: 13px;
    color: #8a7b70;
    margin-top: 6px;
    font-weight: 300;
    letter-spacing: 0.2px;
}

/* Cards row */
.card-row {
    display: flex;
    gap: 16px;
    margin-bottom: 2rem;
}
.info-card {
    flex: 1;
    background: #fff;
    border: 1px solid #e2dbd0;
    border-radius: 8px;
    padding: 18px 20px;
    border-top: 3px solid #9c7c5a;
}
.info-card-label {
    font-family: 'Lato', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #b8a898;
    margin-bottom: 6px;
}
.info-card-value {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    color: #3d3530;
    line-height: 1;
}
.info-card-value.accent { color: #6b8e6b; }
.info-card-value.warm   { color: #9c7c5a; }

/* Section heading */
.sec-head {
    font-family: 'Playfair Display', serif;
    font-size: 14px;
    font-weight: 500;
    color: #3d3530;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e2dbd0;
}

/* Image caption */
.img-caption {
    font-size: 11px;
    color: #b8a898;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Table */
.stDataFrame {
    border: 1px solid #e2dbd0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* Upload */
[data-testid="stFileUploaderDropzone"] {
    background: #fff !important;
    border: 1px dashed #c8bfb4 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #9c7c5a !important;
}

/* Download button */
.stDownloadButton button {
    background: transparent !important;
    color: #3d3530 !important;
    border: 1.5px solid #3d3530 !important;
    border-radius: 6px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.18s;
}
.stDownloadButton button:hover {
    background: #3d3530 !important;
    color: #faf7f2 !important;
}

/* Divider */
.warm-div {
    border: none;
    border-top: 1px solid #e2dbd0;
    margin: 1.8rem 0;
}

/* Progress */
.stProgress > div > div {
    background: #9c7c5a !important;
}

/* Success */
.stAlert {
    border-radius: 8px !important;
    border-left: 3px solid #6b8e6b !important;
    background: #f0f7f0 !important;
    font-family: 'Lato', sans-serif !important;
}

/* Hint text */
.hint-text {
    font-size: 12px;
    color: #b8a898;
    font-weight: 300;
    letter-spacing: 0.3px;
    margin-top: -6px;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO("best.pt")


def detect_image(image, model, fname="detected"):
    results = model(image)
    ann = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    os.makedirs("output", exist_ok=True)
    out = os.path.join("output", f"{fname}_detected.jpg")
    cv2.imwrite(out, cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
    return ann, results[0], out


def detect_video(video_path, model, pb=None):
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps  = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs("output", exist_ok=True)
    base = os.path.basename(video_path).split('.')[0]
    out_path = os.path.join("output", f"{base}_detected.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        out.write(model(frame)[0].plot())
        i += 1
        if pb and total > 0: pb.progress(min(i / total, 1.0))
    cap.release(); out.release()
    return out_path


def make_df(results):
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame()
    data  = boxes.data.cpu().numpy()
    names = results.names
    rows  = []
    for r in data:
        x1, y1, x2, y2, conf, cls = r
        rows.append({"Item": names[int(cls)], "Confidence": f"{conf:.1%}",
                     "X1": int(x1), "Y1": int(y1), "X2": int(x2), "Y2": int(y2)})
    return pd.DataFrame(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Mode")
    input_type = st.radio("", ["Image", "Video"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Settings")
    conf = st.slider("Confidence threshold", 0.10, 0.95, 0.40, 0.05)
    st.markdown("---")
    st.markdown(
        "<p style='font-size:12px;color:#b8a898;font-weight:300;'>Place best.pt in the same directory.</p>",
        unsafe_allow_html=True,
    )

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <p class="page-eyebrow">Safety Intelligence</p>
  <h1 class="page-title">PPE Detection</h1>
  <p class="page-desc">Upload an image or video — the model will identify protective equipment in the scene.</p>
</div>
""", unsafe_allow_html=True)

# ── Image ─────────────────────────────────────────────────────────────────────
if input_type == "Image":
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    st.markdown('<p class="hint-text">Accepts JPG, JPEG or PNG files</p>', unsafe_allow_html=True)

    if uploaded:
        image = Image.open(uploaded)
        fname = os.path.splitext(uploaded.name)[0]

        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.markdown('<p class="img-caption">Original</p>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)

        with st.spinner("Analysing…"):
            ann, results, out_path = detect_image(image, model, fname)

        df = make_df(results)
        n  = len(df)
        nc = df["Item"].nunique() if not df.empty else 0

        st.markdown(f"""
        <div class="card-row">
          <div class="info-card">
            <div class="info-card-label">Total detections</div>
            <div class="info-card-value accent">{n}</div>
          </div>
          <div class="info-card">
            <div class="info-card-label">Unique items</div>
            <div class="info-card-value">{nc}</div>
          </div>
          <div class="info-card">
            <div class="info-card-label">Confidence filter</div>
            <div class="info-card-value warm">{conf:.0%}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with col_b:
            st.markdown('<p class="img-caption">Annotated</p>', unsafe_allow_html=True)
            st.image(ann, use_column_width=True)

        if not df.empty:
            st.markdown('<hr class="warm-div">', unsafe_allow_html=True)
            st.markdown('<p class="sec-head">Detection log</p>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

        with open(out_path, "rb") as f:
            st.download_button("Save result", f, file_name=os.path.basename(out_path), mime="image/jpeg")

# ── Video ─────────────────────────────────────────────────────────────────────
elif input_type == "Video":
    uploaded_v = st.file_uploader("Upload video", type=["mp4","mov","avi"], label_visibility="collapsed")
    st.markdown('<p class="hint-text">Accepts MP4, MOV or AVI files</p>', unsafe_allow_html=True)

    if uploaded_v:
        st.markdown('<p class="sec-head">Source video</p>', unsafe_allow_html=True)
        st.video(uploaded_v)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_v.read())
            tmp_path = tmp.name

        st.markdown('<hr class="warm-div">', unsafe_allow_html=True)
        st.markdown('<p class="sec-head">Processing</p>', unsafe_allow_html=True)
        pb = st.progress(0)

        with st.spinner("Running detection frame by frame…"):
            out_v = detect_video(tmp_path, model, pb)

        st.success("Done — detection complete.")
        st.markdown('<p class="sec-head">Processed output</p>', unsafe_allow_html=True)
        st.video(out_v)

        with open(out_v, "rb") as f:
            st.download_button("Save result", f, file_name=os.path.basename(out_v), mime="video/mp4")
