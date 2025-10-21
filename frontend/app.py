# frontend/app.py
import streamlit as st
import requests
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="🧠 SecureWatch Multimodal Dashboard", layout="wide")

st.title("🧠 SecureWatch Multimodal Dashboard")
st.markdown("Use the sidebar to pick a feature. The app calls backend FastAPI services running on the same machine (localhost).")

# --- Backend base URLs (editable in sidebar) ---
st.sidebar.header("Backend configuration")
host = st.sidebar.text_input("Backend host", value="http://localhost")
sentiment_port = st.sidebar.text_input("Sentiment port", value="8001")
textgen_port = st.sidebar.text_input("TextGen port", value="8002")
imggen_port = st.sidebar.text_input("ImageGen port", value="8003")
seg_port = st.sidebar.text_input("Segmentation port", value="8004")

BASE_SENTIMENT = f"{host}:{sentiment_port}"
BASE_TEXTGEN = f"{host}:{textgen_port}"
BASE_IMGEN = f"{host}:{imggen_port}"
BASE_SEG = f"{host}:{seg_port}"

st.sidebar.markdown("---")
st.sidebar.markdown("Example: If you run services in Colab, keep host as http://localhost (Streamlit server makes server-side requests).")

# --- Feature selector ---
feature = st.sidebar.selectbox("Select a Feature", ["Sentiment Analysis", "RAG / Text Generation", "Text → Image", "Text-based Segmentation"])

# --- Sentiment Analysis ---
if feature == "Sentiment Analysis":
    st.header("📝 Sentiment Analysis")
    text = st.text_area("Enter text to analyze", height=150)
    if st.button("Analyze Sentiment"):
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            try:
                with st.spinner("Analyzing..."):
                    resp = requests.post(f"{BASE_SENTIMENT}/predict", json={"text": text}, timeout=30)
                if resp.status_code == 200:
                    j = resp.json()
                    st.success(f"Sentiment: **{j.get('sentiment')}**  (score: {j.get('score', j.get('confidence', 'n/a'))})")
                    if "probabilities" in j:
                        st.json(j["probabilities"])
                else:
                    st.error(f"Service error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# --- RAG / Text Generation ---
elif feature == "RAG / Text Generation":
    st.header("🧾 RAG / Text Generation")
    query = st.text_area("Ask a question (RAG):", height=150, placeholder="e.g., How to detect SSH brute force attacks?")
    if st.button("Generate Answer"):
        if not query.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Generating..."):
                    resp = requests.post(f"{BASE_TEXTGEN}/generate", json={"query": query}, timeout=60)
                if resp.status_code == 200:
                    answer = resp.json().get("answer") or resp.json().get("result") or resp.text
                    st.markdown("**Answer:**")
                    st.write(answer)
                else:
                    st.error(f"Service error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# --- Text -> Image Generation ---
elif feature == "Text → Image":
    st.header("🖼️ Text → Image Generation")
    prompt = st.text_area("Enter an image prompt:", height=150, placeholder="e.g., A futuristic SOC dashboard, photorealistic")
    max_preview = st.slider("Preview width (px)", 200, 1200, 600)
    if st.button("Generate Image"):
        if not prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            try:
                with st.spinner("Generating image (this may take a while)..."):
                    resp = requests.post(f"{BASE_IMGEN}/generate-image", json={"prompt": prompt}, timeout=120)
                if resp.status_code == 200:
                    img = Image.open(BytesIO(resp.content))
                    st.image(img, caption="Generated image", use_column_width=True, width=max_preview)
                    # offer download
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button("Download image (PNG)", data=buf, file_name="generated.png", mime="image/png")
                else:
                    st.error(f"Service error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# --- Text-based Segmentation ---
elif feature == "Text-based Segmentation":
    st.header("🔍 Text-based Image Segmentation (fallback)")
    uploaded = st.file_uploader("Upload an image", accept_multiple_files=False, type=["png", "jpg", "jpeg"])
    seg_prompt = st.text_input("What to segment? (short text prompt)", placeholder="e.g., person, car, dog")
    cluster_k = st.slider("Mask overlay opacity (%)", 10, 100, 60)
    if st.button("Segment Image"):
        if uploaded is None:
            st.warning("Upload an image first.")
        elif not seg_prompt.strip():
            st.warning("Provide a short segmentation prompt.")
        else:
            try:
                with st.spinner("Sending image to segmentation service..."):
                    # Streamlit's uploaded file -> bytes
                    file_bytes = uploaded.read()
                    files = {"image": (uploaded.name, file_bytes, uploaded.type)}
                    data = {"prompt": seg_prompt}
                    resp = requests.post(f"{BASE_SEG}/segment-image/", files=files, data=data, timeout=120)
                if resp.status_code == 200:
                    img = Image.open(BytesIO(resp.content))
                    st.image(img, caption="Segmented result", use_column_width=True)
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button("Download segmented image (PNG)", data=buf, file_name="segmented.png", mime="image/png")
                else:
                    st.error(f"Service error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ — calls backend FastAPI services on the server (localhost).")
