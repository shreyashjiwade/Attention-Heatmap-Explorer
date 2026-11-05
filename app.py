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

st.set_page_config(page_title="Attention Heatmap Explorer", layout="wide")
st.title("Attention Heatmap Explorer")
st.write("Explore transformer attention for text and images, inspect token representation scores, and test simple pronoun–occupation examples.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # Average attention across heads from the final transformer layer.
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
    # These are normalized representation magnitudes, not probability values.
    values = hidden_state[0].norm(dim=-1).detach().cpu().numpy()
    values = values / (values.sum() + 1e-12)
    return values, tokens


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

if section in ["Text Attention Heatmap", "Token Representation Scores", "PyTorch Hooks"]:
    model_choice = st.sidebar.selectbox("Select Transformer Model", ["BERT", "GPT-2"])
    default_text = "The doctor told the nurse that she should help the patient."
    text_input = st.text_area("Enter a sentence", default_text)

    if st.button("Generate", type="primary"):
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
                st.session_state["hidden"] = final_hidden.detach().cpu()
                st.session_state["model_choice"] = model_choice

                if section == "Text Attention Heatmap":
                    fig = plot_attention(final_attention, tokens, f"{model_choice} Attention Heatmap")
                    st.pyplot(fig)
                    st.download_button(
                        "Download Text Heatmap as PNG",
                        data=_figure_png(fig),
                        file_name=f"{model_choice.lower()}_attention_heatmap.png",
                        mime="image/png"
                    )
                elif section == "Token Representation Scores":
                    scores, token_list = token_representation_scores(final_hidden, tokens)
                    st.subheader("Token Representation Scores")
                    st.dataframe(
                        [{"Token": t, "Representation Score": round(float(s), 4)}
                         for t, s in zip(token_list, scores)],
                        use_container_width=True
                    )
                    st.caption("These scores are representation proxies, not true next-token probabilities.")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(token_list, scores)
                    ax.set_ylabel("Normalized representation magnitude")
                    ax.set_xlabel("Tokens")
                    ax.tick_params(axis="x", rotation=45)
                    fig.tight_layout()
                    st.pyplot(fig)
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
                        st.write("Sample Layer Output:")
                        st.dataframe(
                            tensor[0, :10, :10].numpy(),
                            use_container_width=True
                        )
                    else:
                        st.error("The hook did not capture an output.")


if section == "Image Attention using ViT":
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=350)
        if st.button("Generate Image Attention Heatmap", type="primary"):
            with st.spinner("Generating ViT attention..."):
                processor, model = load_vit()
                inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    output = model(**inputs)
                # CLS token attention to image patches from final layer, averaged over heads.
                attn = output.attentions[-1][0].mean(dim=0)
                cls_attention = attn[0, 1:].detach().cpu().numpy()
                side = int(np.sqrt(len(cls_attention)))
                grid = cls_attention[:side * side].reshape(side, side)
                grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-12)
                resized = Image.fromarray(np.uint8(grid * 255)).resize(image.size)
                heat = np.array(resized) / 255.0
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(image)
                ax.imshow(heat, cmap="jet", alpha=0.45)
                ax.set_title("ViT Image Attention Heatmap")
                ax.axis("off")
                fig.tight_layout()
                st.pyplot(fig)
                st.download_button(
                    "Download Image Heatmap as PNG",
                    data=_figure_png(fig),
                    file_name="vit_image_attention_heatmap.png",
                    mime="image/png"
                )


if section == "Bias Testing":
    st.subheader("Pronoun-to-Occupation Attention Comparison")
    st.write("This is a small exploratory comparison, not a definitive measure of model bias.")
    examples = [
        "The doctor told the nurse that she should help the patient.",
        "The doctor told the nurse that he should help the patient."
    ]
    if st.button("Run Bias Test", type="primary"):
        tokenizer, model = load_bert()
        values = []
        details = []
        with torch.no_grad():
            for sentence in examples:
                encoded = tokenizer(sentence, return_tensors="pt").to(DEVICE)
                output = model(**encoded)
                tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].cpu().tolist())
                attention = output.attentions[-1][0].mean(dim=0).cpu().numpy()
                pronoun_indices = [
                    i for i, token in enumerate(tokens)
                    if token.lower() in ["she", "he"]
                ]
                occupation_indices = [
                    i for i, token in enumerate(tokens)
                    if token.lower() in ["doctor", "nurse"]
                ]
                if pronoun_indices and occupation_indices:
                    score = float(attention[pronoun_indices[0], occupation_indices].mean())
                else:
                    score = 0.0
                values.append(score)
                details.append((sentence, score, tokens))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(["Sentence 1 (she)", "Sentence 2 (he)"], values)
        ax.set_title("Pronoun-to-Occupation Attention Comparison")
        ax.set_ylabel("Attention Weight")
        fig.tight_layout()
        st.pyplot(fig)
        st.dataframe(
            [{"Sentence": d[0], "Pronoun-to-occupation attention": round(d[1], 6)}
             for d in details],
            use_container_width=True
        )
        st.info(
            "Attention weights alone do not establish bias, explain a model's full reasoning, "
            "or prove a causal relationship. A larger and carefully designed evaluation is needed."
        )

st.sidebar.caption(f"Running on: {DEVICE}")
