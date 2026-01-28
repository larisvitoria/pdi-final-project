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

st.set_page_config(
    page_title="CBIS-DDSM Composite Inference",
    page_icon=":material/conversion_path:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    
    .stApp { background-color: #f4f7f9; }
            
    section[data-testid="stMain"] input:focus,
    section[data-testid="stMain"] div[data-baseweb="input"]:focus-within,
    section[data-testid="stMain"] div[data-baseweb="select"] > div:focus-within {
        border-color: #001f3f !important;
        box-shadow: 0 0 0 2px rgba(0, 31, 63, 0.4) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] { 
        background-color: #001f3f !important; 
    }

    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] [data-testid="stHeaderElement"] {
        color: white !important;
    }
            
    section[data-testid="stSidebar"] [data-testid="stIconMaterial"] {
        color: white !important;
    }
            
    section[data-testid="stSidebar"] p {
        color: white !important;
    }
            
    section[data-testid="stSidebar"] hr {
        border-top: 1px solid #ffffff !important;
        opacity: 0.3; /* Optional: makes it look cleaner/subtle */
    }
            
    section[data-testid="stSidebar"] input:focus,
    section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
        border-color: #89CFF0 !important;
        box-shadow: 0 0 0 2px rgba(137, 207, 240, 0.4) !important;
    }
    
    /* Input fields in sidebar */
    section[data-testid="stSidebar"] input[type="text"],
    section[data-testid="stSidebar"] input[type="number"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #1A1A1A !important;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
    }
    
    section[data-testid="stSidebar"] input[type="text"]::placeholder {
        color: #666666 !important;
    }
            
    /* Slider styling */
    section[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {
        background-color: #5BA3F5 !important;
    }
            
    section[data-testid="stSidebar"] [data-baseweb="slider"] div[class*="InnerTrack"] {
        background-color: #89CFF0 !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        background-color: #003366;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00509d;
        border: none;
        color: white;
    }

    /* Expander styling */
    [data-testid="stExpander"] {
        border: 1px solid #d1d9e6;
        border-radius: 8px;
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.image("logo_ufc.svg", width=250)

st.title("CBIS-DDSM Composite Inference")
st.caption(":material/query_stats: Segmentation → ROI crop → Classification")
st.divider()

with st.sidebar:
    st.header(":material/settings: **System Config**")

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

st.subheader("Input Image", divider="gray")
uploaded = st.file_uploader("Upload mammogram (jpg/png)", type=["jpg", "jpeg", "png"])
image_path = st.text_input("Or provide image path", value="")
if uploaded:
    col_l, col_img, col_r = st.columns([0.5, 1, 0.5])
    
    with col_img:
        st.image(uploaded, caption="Preview", use_container_width=True)

output_container = st.empty()

run = st.button("Run Inference", type="primary", icon=":material/play_arrow:")

st.subheader("Output", divider="gray")

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
                
                tab_visual, tab_logs = st.tabs([
                    ":material/visibility: Visual Results", 
                    ":material/description: Output Console"
                ])

                with tab_visual:
                    if overlay_data or os.path.isfile(image_file):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if os.path.isfile(image_file):
                                st.subheader("Original Image")
                                st.image(image_file, use_container_width=True)
                        
                        with col2:
                            if overlay_data:
                                st.subheader("Segmentation (ROI)")
                                st.image(overlay_data, use_container_width=True)
                        
                        st.success("Analysis complete.", icon=":material/check_circle:")
                    else:
                        st.warning("Inferência finalizada, mas nenhuma imagem foi gerada.")

                with tab_logs:
                    st.markdown("Process output logs")
                    st.code(output_text, language="bash")

            except Exception as exc:
                st.error(f"Error during inference: {str(exc)}", icon=":material/error:")
