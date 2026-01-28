import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st


DEFAULT_SEG_MODEL = "deeplabv3p"
DEFAULT_SEG_CKPT = "seg_runs_mass_new/deeplabv3p/deeplabv3p_best.pt"
DEFAULT_CLF_MODELS = "resnext50_32x4d densenet121 efficientnet_v2_s"
DEFAULT_CLF_CKPTS = "resnext50_32x4d_best.pth densenet121_best.pth efficientnet_v2_s_best.pth"


st.set_page_config(page_title="CBIS-DDSM Composite Inference", layout="wide")

st.title("CBIS-DDSM Composite Inference")
st.caption("Segmentation → ROI crop → Classification")

with st.sidebar:
    st.header("Models")
    seg_models = st.text_input("Seg models (space-separated)", value=DEFAULT_SEG_MODEL)
    seg_ckpts = st.text_input("Seg checkpoints (space-separated)", value=DEFAULT_SEG_CKPT)
    seg_threshold = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05)

    st.divider()
    clf_models = st.text_input("Classifier models (space-separated)", value=DEFAULT_CLF_MODELS)
    clf_ckpts = st.text_input("Classifier checkpoints (space-separated)", value=DEFAULT_CLF_CKPTS)
    clf_threshold = st.slider("Classifier threshold", 0.3, 0.9, 0.7, 0.01)

    st.divider()
    view = st.selectbox("View", ["CC", "MLO"], index=0)
    crop_size = st.number_input("Crop size", min_value=256, max_value=1200, value=600, step=16)
    final_size = st.number_input("Final size", min_value=256, max_value=1024, value=512, step=16)

    st.divider()
    st.caption("Paths are relative to the project root unless absolute.")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Input Image")
    uploaded = st.file_uploader("Upload mammogram (jpg/png)", type=["jpg", "jpeg", "png"])
    image_path = st.text_input("Or provide image path", value="")

with col_right:
    st.subheader("Output")
    output_container = st.empty()

run = st.button("Run inference", type="primary")


def run_inference(image_file: str) -> tuple[str, str]:
    seg_models_list = seg_models.strip().split()
    seg_ckpts_list = seg_ckpts.strip().split()
    clf_models_list = clf_models.strip().split()
    clf_ckpts_list = clf_ckpts.strip().split()

    if len(seg_models_list) != len(seg_ckpts_list):
        raise ValueError("Seg models and seg ckpts counts do not match.")
    if len(clf_models_list) != len(clf_ckpts_list):
        raise ValueError("Classifier models and ckpts counts do not match.")

    with tempfile.TemporaryDirectory() as tmpdir:
        overlay_path = os.path.join(tmpdir, "overlay.png")
        cmd = [
            sys.executable,
            "single_image_infer.py",
            "--image",
            image_file,
            "--view",
            view,
            "--seg_models",
            *seg_models_list,
            "--seg_ckpts",
            *seg_ckpts_list,
            "--seg_threshold",
            str(seg_threshold),
            "--crop_size",
            str(crop_size),
            "--final_size",
            str(final_size),
            "--clf_models",
            *clf_models_list,
            "--clf_ckpts",
            *clf_ckpts_list,
            "--clf_threshold",
            str(clf_threshold),
            "--overlay_path",
            overlay_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        overlay_data = None
        if os.path.exists(overlay_path):
            with open(overlay_path, "rb") as f:
                overlay_data = f.read()
        return result.stdout + result.stderr, overlay_data


if run:
    if uploaded is None and not image_path:
        st.error("Please upload an image or provide a path.")
    else:
        if uploaded is not None:
            temp_path = Path(".streamlit_tmp")
            temp_path.mkdir(exist_ok=True)
            img_path = temp_path / uploaded.name
            img_path.write_bytes(uploaded.getbuffer())
            image_file = str(img_path)
        else:
            image_file = image_path

        with st.spinner("Running inference..."):
            try:
                output_text, overlay_data = run_inference(image_file)
                output_container.text(output_text)
                if overlay_data:
                    st.image(overlay_data, caption="Segmentation overlay", use_column_width=True)
                if os.path.isfile(image_file):
                    st.image(image_file, caption="Original image", use_column_width=True)
            except Exception as exc:
                st.error(str(exc))
