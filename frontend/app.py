import streamlit as st
import requests

st.title("🧠 SecureWatch Multimodal Dashboard")

st.sidebar.title("Select a Feature")
choice = st.sidebar.selectbox("Feature", ["Sentiment Analysis"])

if choice == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    text = st.text_area("Enter text here:")
    if st.button("Analyze"):
        if text.strip():
            try:
                # Send request to sentiment FastAPI service
                response = requests.post("http://localhost:8001/predict", json={"text": text})
                if response.status_code == 200:
                    st.success(f"Sentiment: {response.json()['sentiment']}")
                else:
                    st.error("Error: could not get sentiment.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text.")
