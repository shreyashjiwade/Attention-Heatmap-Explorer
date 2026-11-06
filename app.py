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

# Dark style for charts to match the new Charcoal background
plt.style.use('dark_background')

# ==========================================
# PAGE CONFIGURATION 
# ==========================================
st.set_page_config(
    page_title="Attention Heatmap Explorer", 
    page_icon="✨",
    layout="wide"
)

# ==========================================
# PREMIUM CHARCOAL GRAY CSS (No Blue, No White)
# ==========================================
st.markdown("""
<style>
    /* Professional Matte Charcoal Gray Background */
    .stApp {
        background-color: #18181B;
        color: #F4F4F5;
    }
    /* Slightly lighter sidebar */
    [data-testid="stSidebar"] {
        background-color: #27272A;
        border-right: 1px solid #3F3F46;
    }
    /* Transparent header */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    /* Clean text input */
    .stTextArea textarea {
        background-color: #27272A !important;
        color: #F4F4F5 !important;
        border: 1px solid #52525B !important;
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
    
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    sns.heatmap(matrix[:n, :n], xticklabels=tokens[:n], yticklabels=tokens[:n], cmap="viridis", ax=ax)
    
    ax.set_title(title, color="#F4F4F5", pad=15, fontweight='bold')
    ax.set_xlabel("Key Tokens", color="#A1A1AA")
    ax.set_ylabel("Query Tokens", color="#A1A1AA")
    plt.xticks(rotation=45, ha="right", color="#E4E4E7")
    plt.yticks(rotation=0, color="#E4E4E7")
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
# MAIN PAGE UI
# ==========================================
st.markdown("""
    <div style='text-align: center; padding: 2rem 0 2rem 0;'>
        <h1 style='font-size: 3.5rem; font-weight: 800; color: #10B981; margin-bottom: 0.5rem;'>
            Attention Heatmap Explorer
        </h1>
        <p style='font-size: 1.2rem; color: #A1A1AA; font-weight: 500; max-width: 800px; margin: 0 auto; line-height: 1.6;'>
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
