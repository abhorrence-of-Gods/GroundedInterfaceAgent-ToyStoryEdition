import streamlit as st
import torch
import numpy as np
from models.transformers import UnifiedLatentWorldTransformer

st.set_page_config(page_title="GIA Attention & HFP Dashboard", layout="wide")

st.title("GIA Transformer Attention & Human Feedback Probabilities")

@st.cache_resource
def load_model():
    model = UnifiedLatentWorldTransformer()
    model.eval()
    return model

model = load_model()

st.sidebar.header("Input Settings")
B = st.sidebar.slider("Batch size", 1, 8, 2)
L = st.sidebar.slider("Sequence length", 3, 10, 5)

if st.sidebar.button("Generate Random Pass"):
    tokens = torch.randn(B, L, model.embed_dim)
    reward_prev = torch.zeros(B, 1)
    with torch.no_grad():
        out = model(tokens, reward_prev)
    hfp = out["hfp_prob"].squeeze(-1).cpu().numpy()
    st.subheader("Human Feedback Probabilities")
    st.write(hfp)

    # show attention weights of last layer head 0 (if available)
    if hasattr(model.encoder.layers[-1], "self_attn"):
        # Not trivial to extract weights without forward hook; placeholder visual
        st.subheader("Attention map (placeholder)")
        attn_placeholder = np.random.rand(L + 2, L + 2)
        st.image(attn_placeholder)
    else:
        st.info("Attention weights unavailable in this minimal example.") 