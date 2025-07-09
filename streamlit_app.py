import streamlit as st
import torch
from model import GPTModel, GPT_CONFIG_124M, generate, text_to_token_ids, token_ids_to_text
import tiktoken
import os 

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load trained model
def load_model():
    ckpt_path = "model_checkpoint.pth"
    print(f"Trying to load checkpoint from: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict' key. Found keys: " + ", ".join(checkpoint.keys()))

    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokens_seen = checkpoint.get("tokens_seen", "?")
    print(f"✅ Loaded model trained on {tokens_seen} tokens")

    return model

if st.sidebar.button("Clear Cache"):
    st.cache_resource.clear()
    st.rerun()

# Title
st.title("LLM Sandbox")

# Sidebar settings
st.sidebar.header("Generation Settings")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
top_k = st.sidebar.slider("Top-k Sampling", min_value=0, max_value=100, value=40, step=5)
max_tokens = st.sidebar.slider("Max New Tokens", min_value=10, max_value=200, value=50, step=10)

# Input area
prompt = st.text_area("Enter your prompt below:", height=150)

# Generate button
if st.button("Generate"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            model = load_model()
            context_size = GPT_CONFIG_124M["context_length"]
            input_ids = text_to_token_ids(prompt, tokenizer).unsqueeze(0)
            input_ids = input_ids[:, -context_size:]  # truncate if needed

            output_ids = generate(
                model=model,
                idx=input_ids,
                max_new_tokens=max_tokens,
                context_size=context_size,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None
            )

            output_text = token_ids_to_text(output_ids[0], tokenizer)
            st.subheader("📝 Output")
            st.write(output_text)

# Footer
st.markdown("---")

