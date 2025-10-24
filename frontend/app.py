import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import uuid  # <-- Import this to create unique session IDs

st.set_page_config(page_title="🤖 CyberGuard AI", layout="wide")

# --- Custom CSS for a professional look ---
st.markdown("""
<style>
/* Main app styling */
.main {
    background-color: #f0f2f6; /* Light gray background */
    padding-top: 0rem; /* Move content to the top */
}

/* Title */
h1 {
    color: #1a1a68; /* Deep blue */
    font-family: 'Arial Black', sans-serif;
    text-align: center;
    margin-bottom: 0;
    padding-top: 0;
}

/* Headers (h2) for features */
h2 {
    color: #333;
    font-family: 'Arial', sans-serif;
    text-align: center;
}

/* Sidebar styling */
.stSidebar {
    background-color: #ffffff;
    border-right: 2px solid #e0e0e0;
}

/* Buttons */
.stButton>button {
    background-color: #1a1a68; /* Deep blue */
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    width: 100%;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #3e3e8f; /* Lighter blue on hover */
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.stButton>button:active {
    background-color: #10104a; /* Darker blue on click */
}

/* Text area & inputs */
.stTextArea textarea, .stTextInput input {
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    background-color: #ffffff;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #1a1a68;
    box-shadow: 0 0 0 2px #3e3e8f30;
}

/* Info/Success/Error boxes */
.stSuccess {
    background-color: #e6f7ff;
    border: 1px solid #b3e0ff;
    border-radius: 8px;
    color: #0056b3;
}
.stError {
    background-color: #ffe6e6;
    border: 1px solid #ffb3b3;
    border-radius: 8px;
    color: #b30000;
}
.stWarning {
    background-color: #fffbe6;
    border: 1px solid #ffe58f;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# --- Main Title ---
st.title("🤖 CyberGuard AI")
st.markdown("<h3 style='text-align: center; color: #333; font-family: Arial, sans-serif; font-weight: normal; margin-top: 0; padding-top: 0;'>Multimodal Dashboard <span style='font-size: 0.9em; color: #555;'>&nbsp;&nbsp;|&nbsp;&nbsp;Use the sidebar to pick a feature.</span></h3>", unsafe_allow_html=True)


# --- Sidebar ---
st.sidebar.header("Select a Feature")
feature = st.sidebar.selectbox(
    "Select a Feature", 
    ["Chat with CyberGuard AI", "Sentiment Analysis", "Text → Image", "Text-based Segmentation"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.header("Backend Configuration")
# *** UPDATED: Set the default host to your ngrok URL (or localhost if local) ***
# *** You can get this from your Colab server cell ***
ngrok_url = st.sidebar.text_input("Backend URL", value="http://YOUR_NGROK_URL_HERE.io")
st.sidebar.info("Paste your ngrok URL from Colab here. If running locally, use http://localhost:8002")

# *** UPDATED: No need for separate ports if ngrok is used ***
BASE_TEXTGEN = ngrok_url
BASE_SENTIMENT = ngrok_url.replace("8002", "8001") # Assuming other services run on different ports
BASE_IMGEN = ngrok_url.replace("8002", "8003")
BASE_SEG = ngrok_url.replace("8002", "8004")

st.sidebar.markdown("---")
st.sidebar.markdown("This Streamlit app sends requests to the FastAPI backend running in your Colab notebook.")

# --- RAG / Text Generation (Renamed to CyberGuard AI) ---
if feature == "Chat with CyberGuard AI":
    st.header("💬 Chat with CyberGuard AI (RAG)")
    st.markdown("<p style='text-align: center;'>Ask a question about your cybersecurity data or general knowledge. The bot has memory.</p>", unsafe_allow_html=True)
    
    # --- [FIX] SESSION ID MANAGEMENT ---
    # Create a unique session ID for this user if it doesn't exist
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Display the session ID (useful for debugging)
    st.sidebar.text(f"Your Session ID: {st.session_state.session_id}")
    
    # --- [FIX] ADD CONVERSATION RESET BUTTON ---
    if st.sidebar.button("Reset Conversation"):
        try:
            # Call the /reset endpoint
            # We pass session_id as a query parameter
            reset_url = f"{BASE_TEXTGEN}/reset"
            resp = requests.post(reset_url, params={"session_id": st.session_state.session_id}, timeout=30)
            
            if resp.status_code == 200:
                st.success("Conversation history has been reset.")
                # Create a new session ID to start a fresh conversation
                st.session_state.session_id = str(uuid.uuid4())
                st.sidebar.text(f"New Session ID: {st.session_state.session_id}")
                st.rerun() # Rerun the app to clear the page
            else:
                st.error(f"Error resetting conversation: {resp.text}")
        except Exception as e:
            st.error(f"Failed to connect to reset endpoint: {e}")

    query = st.text_area("Your Question:", height=150, placeholder="e.g., What is a SYN flood attack?  OR  What was the response for the DDoS attack?")
    
    if st.button("Generate Answer"):
        if not query.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("CyberGuard AI is thinking..."):
                    
                    # --- [FIX] UPDATED API CALL ---
                    # 1. Use the correct /chat endpoint
                    # 2. Send the correct JSON payload
                    api_url = f"{BASE_TEXTGEN}/chat"
                    payload = {
                        "question": query,
                        "session_id": st.session_state.session_id
                    }
                    
                    # Increased timeout for slow LlamaCpp model
                    resp = requests.post(api_url, json=payload, timeout=180) 
                
                if resp.status_code == 200:
                    # --- [FIX] UPDATED RESPONSE HANDLING ---
                    # The backend returns "answer" and "source_documents"
                    answer = resp.json().get("answer", "No answer found.")
                    sources = resp.json().get("source_documents", [])
                    
                    st.markdown("### 🤖 CyberGuard AI's Answer:")
                    st.info(answer)
                    
                    # Display the source documents
                    if sources:
                        with st.expander("Show Sources"):
                            st.json(sources)
                else:
                    st.error(f"Service error: {resp.status_code} - {resp.text}")
            except requests.exceptions.Timeout:
                st.error("The request timed out. The server (LlamaCpp) might be too slow or still loading.")
            except Exception as e:
                st.error(f"Request failed: {e}")

# --- Sentiment Analysis (No changes needed, assuming endpoint is correct) ---
elif feature == "Sentiment Analysis":
    st.header("📝 Sentiment Analysis")
    st.markdown("<p style='text-align: center;'>Analyze the sentiment of a piece of text.</p>", unsafe_allow_html=True)
    
    text = st.text_area("Enter text to analyze:", height=150, placeholder="e.g., 'The system is down again, this is frustrating.'")
    
    if st.button("Analyze Sentiment"):
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            try:
                with st.spinner("Analyzing..."):
                    resp = requests.post(f"{BASE_SENTIMENT}/predict", json={"text": text}, timeout=30)
                if resp.status_code == 200:
                    j = resp.json()
                    st.success(f"Sentiment: **{j.get('sentiment')}** (Score: {j.get('score', j.get('confidence', 'n/a'))})")
                    if "probabilities" in j:
                        st.json(j["probabilities"])
                else:
                    st.error(f"Service error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# --- Text -> Image Generation (No changes needed, assuming endpoint is correct) ---
elif feature == "Text → Image":
    st.header("🖼️ Text → Image Generation")
    st.markdown("<p style='text-align: center;'>Generate an image from a text prompt.</p>", unsafe_allow_html=True)
    
    prompt = st.text_area("Enter an image prompt:", height=150, placeholder="e.g., A futuristic SOC dashboard, photorealistic")
    
    if st.button("Generate Image"):
        if not prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            try:
                with st.spinner("Generating image (this may take a while)..."):
                    resp = requests.post(f"{BASE_IMGEN}/generate-image", json={"prompt": prompt}, timeout=120)
                if resp.status_code == 200:
                    img = Image.open(BytesIO(resp.content))
                    st.image(img, caption="Generated image", use_column_width=True)
                    # offer download
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button("Download image (PNG)", data=buf, file_name="generated.png", mime="image/png")
                else:
                    st.error(f"Service error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# --- Text-based Segmentation (No changes needed, assuming endpoint is correct) ---
elif feature == "Text-based Segmentation":
    st.header("🔍 Text-based Image Segmentation")
    st.markdown("<p style='text-align: center;'>Upload an image and identify objects to segment.</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload an image", accept_multiple_files=False, type=["png", "jpg", "jpeg"])
    seg_prompt = st.text_input("What to segment?", placeholder="e.g., person, car, dog")
    
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
st.markdown("<p style='text-align: center; color: #555;'>Made with ❤️ — calls backend FastAPI services on the server.</p>", unsafe_allow_html=True)
