from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment Analysis Service")

# Load pretrained DistilBERT model
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TextInput):
    result = sentiment_model(data.text)[0]
    return {"sentiment": result["label"], "score": result["score"]}
