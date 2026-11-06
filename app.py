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
# FAINT & CLEAN MINIMALIST CSS
# ==========================================
st.markdown("""
<style>
    /* Ultra Faint / Clean Minimalist Background */
    .stApp {
        background-color: #FCFCFD;
        color: #111827;
    }
    /* Faint Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F3F4F6;
        border-right: 1px solid #E5E7EB;
    }
    /* Transparent header */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    /* Clean white text input */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 8px;
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
    model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True, output_hidden_states=True).to(DEVICE).eval()
    return tokenizer, model

@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_attentions=True, output_hidden_states=True).to(DEVICE).eval()
    return tokenizer, model

@st.cache_resource
def load_vit():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=True).to(DEVICE).eval()
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
    
    # Transparent plot background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    sns.heatmap(matrix[:n, :n], xticklabels=tokens[:n], yticklabels=tokens[:n], cmap="viridis", ax=ax)
    
    # Dark text for faint theme
    ax.set_title(title, color="#111827", pad=15, fontweight='bold')
    ax.set_xlabel("Key Tokens", color="#4B5563")
    ax.set_ylabel("Query Tokens", color="#4B5563")
    plt.xticks(rotation=45, ha="right", color="#
