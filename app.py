import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Attention Heatmap Explorer | HW 33",
    page_icon="🔍",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #161b22;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("### HW 33: Attention Mechanisms")
model_type = st.sidebar.selectbox(
    "Choose Model Type",
    ["BERT (Text)", "GPT-2 (Text)", "Vision Transformer (ViT - Image)"]
)

st.sidebar.markdown("---")
st.sidebar.info("Repository: [shreyashjiwade/attention-heatmap-explorer](https://github.com/shreyashjiwade/attention-heatmap-explorer)")

# --- Main App Title ---
st.title("🔍 Attention Heatmap Explorer (HW 33)")
st.markdown("A comprehensive Streamlit application to visualize attention weights and token/patch interactions for Transformer models.")

# ==========================================
# 1. BERT / GPT-2 TEXT EXPLORER
# ==========================================
if "Text" in model_type:
    model_name = "bert-base-uncased" if "BERT" in model_type else "gpt2"
    
    @st.cache_resource
    def load_text_model(name):
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name, output_attentions=True)
        model.eval()
        return tokenizer, model

    with st.spinner(f"Loading {model_name} model and tokenizer..."):
        tokenizer, model = load_text_model(model_name)

    user_input = st.text_area("Enter input text for analysis:", "The quick brown fox jumps over the lazy dog.")

    if st.button("Generate Attention Heatmap"):
        if user_input.strip() == "":
            st.warning("Please enter valid text input.")
        else:
            inputs = tokenizer(user_input, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            attentions = outputs.attentions # Tuple of layers: (batch, heads, seq_len, seq_len)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            st.success("Attention weights computed successfully!")
            
            # Layer & Head selection
            layer_idx = st.slider("Select Layer", 0, len(attentions)-1, len(attentions)-1)
            layer_attention = attentions[layer_idx][0].mean(dim=0).cpu().numpy() # Average across heads
            
            # Plot Heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(layer_attention, xticklabels=tokens, yticklabels=tokens, cmap="Blues", ax=ax, cbar=True, linewidths=0.5)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            ax.set_title(f"Attention Heatmap for Layer {layer_idx} ({model_name})", fontsize=12, fontweight='bold')
            st.pyplot(fig)

# ==========================================
# 2. VISION TRANSFORMER (ViT) EXPLORER
# ==========================================
else:
    @st.cache_resource
    def load_vit_model():
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", output_attentions=True)
        model.eval()
        return processor, model

    with st.spinner("Loading Vision Transformer (ViT)..."):
        processor, model = load_vit_model()

    uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image for Analysis", width=350)

        if st.button("Generate ViT Attention Map"):
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            attentions = outputs.attentions # Layers tuple
            # Get last layer attention map for the [CLS] token across patches
            cls_attention = attentions[-1][0, :, 0, 1:].mean(dim=0) # Average across heads
            
            # Reshape to grid (e.g., 14x14 for patch size 16 on 224x224 image)
            grid_size = int(np.sqrt(cls_attention.shape[0]))
            attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
            
            # Normalize map
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image.resize((224, 224)))
            ax.imshow(attention_map, cmap='jet', alpha=0.6, extent=(0, 224, 224, 0))
            ax.axis('off')
            ax.set_title("ViT [CLS] Token Attention Overlay", fontsize=12, fontweight='bold')
            st.pyplot(fig)
            st.success("ViT Attention overlay generated successfully!")
    else:
        st.info("Please upload an image to analyze Vision Transformer attention.")
