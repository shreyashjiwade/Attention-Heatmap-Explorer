import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import (
    AutoTokenizer, AutoModel,
    GPT2Tokenizer, GPT2Model,
    ViTImageProcessor, ViTModel
)

# ==========================================
# PAGE CONFIGURATION 
# ==========================================
st.set_page_config(
    page_title="Attention Heatmap Explorer", 
    page_icon="✨",
    layout="wide"
)

# ==========================================
# PREMIUM SOFT LIGHT THEME CSS
# ==========================================
st.markdown("""
<style>
    /* Clean Mac-like Soft Gradient Background */
    .stApp {
        background-color: #F8FAFC;
        background-image: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
    }
    /* Transparent header */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    /* Clean Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# CACHED FUNCTIONS
# ==========================================
@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained(
        "bert-base-uncased", output_attentions=True, output_hidden_states=True
    ).to(DEVICE).eval()
    return tokenizer, model

@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained(
        "gpt2", output_attentions=True, output_hidden_states=True
    ).to(DEVICE).eval()
    return tokenizer, model

@st.cache_resource
def load_vit():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained(
        "google/vit-base-patch16-224", output_attentions=True
    ).to(DEVICE).eval()
    return processor, model

def _figure_png(fig):
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    buffer.seek(0)
    return buffer.getvalue()

def plot_attention(attention, tokens, title):
    matrix = attention[0].mean(dim=0).detach().cpu().numpy()
    n = min(len(tokens), matrix.shape[0], 32)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.55), max(5, n * 0.45)))
    
    # Transparent background for graph
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    sns.heatmap(
        matrix[:n, :n],
        xticklabels=tokens[:n],
        yticklabels=tokens[:n],
        cmap="viridis",
        ax=ax
    )
    ax.set_title(title, color="#1E293B", pad=15)
    ax.set_xlabel("Key Tokens", color="#475569")
    ax.set_ylabel("Query Tokens", color="#475569")
    plt.xticks(rotation=45, ha="right", color="#334155")
    plt.yticks(rotation=0, color="#334155")
    fig.tight_layout()
    return fig

def token_representation_scores(hidden_state, tokens):
    values = hidden_state[0].norm(dim=-1).detach().cpu().numpy()
    values = values / (values.sum() + 1e-12)
    return values, tokens

# ==========================================
# SIDEBAR UI
# ==========================================
st.sidebar.header("Explorer")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Text Attention Heatmap",
        "Token Representation Scores",
        "Image Attention using ViT",
        "PyTorch Hooks",
        "Bias Testing"
    ]
)

# ==========================================
# MAIN PAGE UI (ATTRACTIVE HEADER)
# ==========================================
st.markdown("""
    <div style='text-align: center; padding: 2rem 0 2rem 0;'>
        <h1 style='font-size: 3.5rem; font-weight: 800; background: -webkit-linear-gradient(45deg, #FF4B4B, #6B46C1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;'>
            Attention Heatmap Explorer
        </h1>
        <p style='font-size: 1.2rem; color: #475569; font-weight: 400; max-width: 800px; margin: 0 auto; line-height: 1.6;'>
            Explore transformer attention for text and images, inspect token representation scores, and test simple pronoun–occupation examples.
        </p>
    </div>
""", unsafe_allow_html=True)
st.write("---")

# ==========================================
# TEXT / TOKEN / HOOKS SECTIONS
# ==========================================
if section in ["Text Attention Heatmap", "Token Representation Scores", "PyTorch Hooks"]:
    model_choice = st.sidebar.selectbox("Select Transformer Model", ["BERT", "GPT-2"])
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        default_text = "The doctor told the nurse that she should help the patient."
        text_input = st.text_area("✍️ Enter a sentence", default_text)

    with col2:
        st.write("") 
        st.write("")
        generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)

    if generate_btn:
