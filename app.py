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
    page_icon="🔍",
    layout="wide"
)

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
    sns.heatmap(
        matrix[:n, :n],
        xticklabels=tokens[:n],
        yticklabels=tokens[:n],
        cmap="viridis",
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
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
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🔍 Attention Heatmap Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7F8C8D;'>Explore transformer attention for text and images, inspect token representation scores, and test simple pronoun–occupation examples.</p>", unsafe_allow_html=True)
st.divider()

# ==========================================
# TEXT / TOKEN / HOOKS SECTIONS
# ==========================================
if section in ["Text Attention Heatmap", "Token Representation Scores", "PyTorch Hooks"]:
    model_choice = st.sidebar.selectbox("Select Transformer Model", ["BERT", "GPT-2"])
    
    st.info("💡 Tip: Try testing different pronouns to observe gender bias in the model.")

    col1, col2 = st.columns([4, 1])
    
    with col1:
        default_text = "The doctor told the nurse that she should help the patient."
        text_input = st.text_area("✍️ Enter a sentence", default_text)

    with col2:
        st.write("") 
        st.write("")
        generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)

    if generate_btn:
        if not text_input.strip():
            st.warning("Please enter a sentence.")
        else:
            with st.spinner("Loading model and generating outputs..."):
                if model_choice == "BERT":
                    tokenizer, model = load_bert()
                    encoded = tokenizer(
                        text_input, return_tensors="pt", truncation=True,
                        max_length=32
                    ).to(DEVICE)
                    with torch.no_grad():
                        output = model(**encoded)
                    tokens = tokenizer.convert_ids_to_tokens(
                        encoded["input_ids"][0].detach().cpu().tolist()
                    )
                    final_attention = output.attentions[-1]
                    final_hidden = output.last_hidden_state
                    hook_layer = model.encoder.layer[-1]
                else:
                    tokenizer, model = load_gpt2()
                    encoded = tokenizer(
                        text_input, return_tensors="pt", truncation=True,
                        max_length=32
                    ).to(DEVICE)
                    with torch.no_grad():
                        output = model(**encoded)
                    tokens = tokenizer.convert_ids_to_tokens(
                        encoded["input_ids"][0].detach().cpu().tolist()
                    )
                    final_attention = output.attentions[-1]
                    final_hidden = output.last_hidden_state
                    hook_layer = model.h[-1]

                st.session_state["tokens"] = tokens
                st.session_state["attention"] = final_attention.detach().cpu()
                st.session_state["hidden"] =
