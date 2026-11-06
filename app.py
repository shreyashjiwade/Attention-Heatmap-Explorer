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

# Dark style for charts to match the new Matte Slate background
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
# PREMIUM MATTE SLATE CSS (Not White, Not Black)
# ==========================================
st.markdown("""
<style>
    /* Professional Matte Slate Background */
    .stApp {
        background-color: #1E293B;
        color: #F8FAFC;
    }
    /* Darker Sidebar for contrast */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #334155;
    }
    /* Transparent header */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    /* Clean text input */
    .stTextArea textarea {
        background-color: #334155 !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
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
    
    ax.set_title(title, color="#F8FAFC", pad=15, fontweight='bold')
    ax.set_xlabel("Key Tokens", color="#94A3B8")
    ax.set_ylabel("Query Tokens", color="#94A3B8")
    plt.xticks(rotation=45, ha="right", color="#CBD5E1")
    plt.yticks(rotation=0, color="#CBD5E1")
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
        <h1 style='font-size: 3.5rem; font-weight: 800; color: #38BDF8; margin-bottom: 0.5rem;'>
            Attention Heatmap Explorer
        </h1>
        <p style='font-size: 1.2rem; color: #94A3B8; font-weight: 500; max-width: 800px; margin: 0 auto; line-height: 1.6;'>
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
        if not text_input.strip():
            st.warning("Please enter a sentence.")
        else:
            with st.spinner("Loading model and generating outputs..."):
                if model_choice == "BERT":
                    tokenizer, model = load_bert()
                    encoded = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=32).to(DEVICE)
                    with torch.no_grad():
                        output = model(**encoded)
                    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].detach().cpu().tolist())
                    final_attention = output.attentions[-1]
                    final_hidden = output.last_hidden_state
                    hook_layer = model.encoder.layer[-1]
                else:
                    tokenizer, model = load_gpt2()
                    encoded = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=32).to(DEVICE)
                    with torch.no_grad():
                        output = model(**encoded)
                    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].detach().cpu().tolist())
                    final_attention = output.attentions[-1]
                    final_hidden = output.last_hidden_state
                    hook_layer = model.h[-1]

                st.session_state["tokens"] = tokens
                st.session_state["attention"] = final_attention.detach().cpu()
                st.session_state["hidden"] = final_hidden.detach().cpu()
                st.session_state["model_choice"] = model_choice

                # --- 1. Text Attention Heatmap ---
                if section == "Text Attention Heatmap":
                    fig = plot_attention(final_attention, tokens, f"{model_choice} Attention Heatmap")
                    st.pyplot(fig)
                    st.download_button("Download Text Heatmap as PNG", data=_figure_png(fig), file_name=f"{model_choice.lower()}_attention_heatmap.png", mime="image/png")
                
                # --- 2. Token Representation Scores ---
                elif section == "Token Representation Scores":
                    scores, token_list = token_representation_scores(final_hidden, tokens)
                    st.subheader("Token Representation Scores")
                    st.dataframe([{"Token": t, "Representation Score": round(float(s), 4)} for t, s in zip(token_list, scores)], use_container_width=True)
                    st.caption("These scores are representation proxies, not true next-token probabilities.")
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    bg_matrix = final_attention[0].mean(dim=0).detach().cpu().numpy()
                    n = len(token_list)
                    bg_matrix = bg_matrix[:n, :n]
                    
                    ax.imshow(bg_matrix, cmap='magma', aspect='auto', zorder=1, alpha=0.4, interpolation='bicubic')
                    ax.bar(token_list, scores, color='#38BDF8', zorder=2, edgecolor='#BAE6FD', linewidth=1)
                    
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#475569')
                    ax.spines['bottom'].set_color('#475569')
                    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0, color='#64748B')
                    
                    ax.set_ylabel("Representation Score", color="#94A3B8", fontsize=10)
                    ax.tick_params(axis="x", rotation=45, colors="#CBD5E1")
                    ax.tick_params(axis="y", colors="#CBD5E1")
                    fig.tight_layout()
                    st.pyplot(fig)
                
                # --- 3. PyTorch Hooks ---
                else:
                    captured = {}
                    def save_output(module, inputs, output):
                        captured["value"] = output[0] if isinstance(output, tuple) else output

                    handle = hook_layer.register_forward_hook(save_output)
                    try:
                        with torch.no_grad():
                            _ = model(**encoded)
                    finally:
                        handle.remove()

                    if "value" in captured:
                        tensor = captured["value"].detach().cpu()
                        st.success("PyTorch Hook captured layer output successfully!")
                        st.write("Captured Layer Output Shape:", tuple(tensor.shape))
                        st.dataframe(tensor[0, :10, :10].numpy(), use_container_width=True)
                    else:
                        st.error("The hook did not capture an output.")

# ==========================================
# ViT IMAGE SECTION
# ==========================================
elif section == "Image Attention using ViT":
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
            
    with col2:
        if uploaded is not None:
            if st.button("🚀 Generate Image Attention Heatmap", type="primary", use_container_width=True):
                with st.spinner("Generating ViT attention..."):
                    processor, model = load_vit()
                    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        output = model(**inputs)
                    
                    attn = output.attentions[-1][0].mean(dim=0)
                    cls_attention = attn[0, 1:].detach().cpu().numpy()
                    side = int(np.sqrt(len(cls_attention)))
                    grid = cls_attention[:side * side].reshape(side, side)
                    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-12)
                    
                    # Correct syntax for image resize
                    resized = Image.fromarray(np.uint8(grid * 255)).resize(image.size)
                    
                    heat = np.array(resized) / 255.0
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    
                    ax.imshow(image)
                    ax.imshow(heat, cmap="magma", alpha=0.45)
                    ax.set_title("ViT Image Attention Heatmap", color="#F8FAFC", fontweight='bold')
                    ax.axis("off")
                    fig.tight_layout()
                    
                    st.pyplot(fig)
                    st.download_button("Download Image Heatmap as PNG", data=_figure_png(fig), file_name="vit_image_attention_heatmap.png", mime="image/png")

# ==========================================
# BIAS TESTING SECTION
# ==========================================
elif section == "Bias Testing":
    st.subheader("Pronoun-to-Occupation Attention Comparison")
    st.write("This is a small exploratory comparison, not a definitive measure of model bias.")
    examples = [
        "The doctor told the nurse that she should help the patient.",
        "The doctor told the nurse that he should help the patient."
    ]
    if st.button("🚀 Run Bias Test", type="primary"):
        with st.spinner("Analyzing bias..."):
            tokenizer, model = load_bert()
            values = []
            details = []
            with torch.no_grad():
                for sentence in examples:
                    encoded = tokenizer(sentence, return_tensors="pt").to(DEVICE)
                    output = model(**encoded)
                    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].cpu().tolist())
                    attention = output.attentions[-1][0].mean(dim=0).cpu().numpy()
                    pronoun_indices = [i for i, token in enumerate(tokens) if token.lower() in ["she", "he"]]
                    occupation_indices = [i for i, token in enumerate(tokens) if token.lower() in ["doctor", "nurse"]]
                    if pronoun_indices and occupation_indices:
                        score = float(attention[pronoun_indices[0], occupation_indices].mean())
                    else:
                        score = 0.0
                    values.append(score)
                    details.append((sentence, score, tokens))
            
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            ax.bar(["Sentence 1 (she)", "Sentence 2 (he)"], values, color=['#818CF8', '#38BDF8'])
            ax.set_title("Pronoun-to-Occupation Attention Comparison", color="#F8FAFC", fontweight='bold')
            ax.set_ylabel("Attention Weight", color="#94A3B8")
            ax.tick_params(colors="#CBD5E1")
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#475569')
            ax.spines['bottom'].set_color('#475569')
            
            fig.tight_layout()
            st.pyplot(fig)
            
            st.dataframe([{"Sentence": d[0], "Pronoun-to-occupation attention": round(d[1], 6)} for d in details], use_container_width=True)
            st.info("Attention weights alone do not establish bias, explain a model's full reasoning, or prove a causal relationship. A larger and carefully designed evaluation is needed.")

st.sidebar.caption(f"Running on: {DEVICE}")
