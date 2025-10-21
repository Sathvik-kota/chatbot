import os
import re
import unicodedata
import string
from typing import Dict
import emoji
import contractions
import pandas as pd
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from contextlib import asynccontextmanager

# -------------------------
# Preprocessing (unchanged)
# -------------------------
class DistilBERTPreprocessor:
    def __init__(self, model_case='uncased'):
        self.model_case = model_case.lower()
        self.slang_dict = self._load_slang_dict()
        self.emoticon_dict = self._load_emoticon_dict()
        self.typo_dict = self._load_typo_dict()

    def _load_slang_dict(self):
        return {'lol': 'laughing out loud', 'lmao': 'laughing my ass off', 'rofl': 'rolling on floor laughing', 'omg': 'oh my god', 'wtf': 'what the fuck', 'brb': 'be right back', 'ttyl': 'talk to you later', 'imo': 'in my opinion', 'imho': 'in my humble opinion', 'fyi': 'for your information', 'btw': 'by the way', 'smh': 'shaking my head', 'tbh': 'to be honest', 'ikr': 'i know right', 'idk': 'i do not know', 'nvm': 'never mind', 'thx': 'thanks', 'u': 'you', 'ur': 'your', 'ppl': 'people', 'plz': 'please', 'w8': 'wait', '2day': 'today', '2morrow': 'tomorrow', '4ever': 'forever', 'b4': 'before', 'c u': 'see you', 'gr8': 'great', 'l8r': 'later', 'm8': 'mate', 'n8': 'night', 'r u': 'are you', 'sup': 'what is up', 'wbu': 'what about you', 'yolo': 'you only live once', 'bae': 'before anyone else', 'fomo': 'fear of missing out', 'rn': 'right now', 'af': 'as fuck', 'ngl': 'not gonna lie', 'periodt': 'period', 'stan': 'support', 'salty': 'bitter', 'lit': 'amazing', 'fire': 'excellent', 'dope': 'cool', 'sick': 'awesome', 'tight': 'cool', 'bomb': 'amazing', 'sus': 'suspicious', 'cap': 'lie', 'no cap': 'no lie', 'bet': 'yes', 'facts': 'truth', 'mood': 'relatable', 'vibe': 'feeling', 'lowkey': 'somewhat', 'highkey': 'definitely', 'deadass': 'seriously', 'fr': 'for real', 'ong': 'on god'}

    def _load_emoticon_dict(self):
        return {':)': 'smile', ':-)': 'smile', '(:': 'smile', ':D': 'big smile', ':-D': 'big smile', 'xD': 'laughing', 'XD': 'laughing', ';)': 'wink', ';-)': 'wink', ':P': 'tongue out', ':-P': 'tongue out', ':p': 'tongue out', ':-p': 'tongue out', ':(': 'sad', ':-(': 'sad', ':/': 'confused', ':-/': 'confused', ':\\': 'confused', ':-\\': 'confused', '>:(': 'angry', '>:-(': 'angry', '<3': 'heart', '</3': 'broken heart', ':|': 'neutral', ':-|': 'neutral', ':o': 'surprised', ':-o': 'surprised', ':O': 'surprised', ':-O': 'surprised'}

    def _load_typo_dict(self):
        return {'gret': 'great', 'awsome': 'awesome', 'recieve': 'receive', 'beutiful': 'beautiful', 'definately': 'definitely', 'seperate': 'separate', 'occured': 'occurred', 'neccessary': 'necessary', 'accomodate': 'accommodate', 'embarass': 'embarrass', 'tommorrow': 'tomorrow', 'begining': 'beginning', 'untill': 'until', 'sucessful': 'successful', 'preffered': 'preferred', 'occassion': 'occasion', 'dissappointed': 'disappointed', 'recomended': 'recommended', 'experiance': 'experience', 'maintainance': 'maintenance', 'enviroment': 'environment', 'goverment': 'government', 'resturant': 'restaurant', 'buisness': 'business', 'reccomend': 'recommend', 'existance': 'existence', 'appearence': 'appearance', 'independant': 'independent', 'persistant': 'persistent', 'concious': 'conscious', 'truely': 'truly', 'wierd': 'weird', 'freind': 'friend', 'thier': 'their', 'recieved': 'received'}

    def preprocess_text(self, text: str) -> str:
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\r\t')
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)
        text = re.sub(r'www\.\S+', '<URL>', text)
        text = re.sub(r'@\w+', '<USER>', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = emoji.demojize(text, delimiters=(" ", " "))
        emoticons_sorted = sorted(self.emoticon_dict.keys(), key=lambda x: -len(x))
        punct_chars = re.escape(string.punctuation)
        for emoticon in emoticons_sorted:
            emotion = self.emoticon_dict[emoticon]
            emot_esc = re.escape(emoticon)
            pattern = re.compile(r'(?<!\S)' + emot_esc + r'(?=(?:\s|$|[' + punct_chars + r']))', flags=re.IGNORECASE)
            text = pattern.sub(' ' + emotion + ' ', text)
        text = contractions.fix(text)
        words = text.split()
        normalized_words = [self.slang_dict.get(word.lower(), word) for word in words]
        text = ' '.join(normalized_words)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'\.{2,}', '.', text)
        if self.model_case == 'uncased':
            text = text.lower()
        words = text.split()
        corrected_words = [self.typo_dict.get(word.lower(), word) for word in words]
        text = ' '.join(corrected_words)
        return re.sub(r'\s+', ' ', text).strip()

# -------------------------
# FastAPI app + startup
# -------------------------
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up Sentiment service...")
    model_path = "./model"  # local model folder (if present)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Preprocessor always available
        ml_models["preprocessor"] = DistilBERTPreprocessor(model_case='uncased')

        # Load tokenizer & model from local folder if available; otherwise fallback to HF SST-2 finetuned model
        if os.path.exists(model_path) and os.listdir(model_path):
            print(f"Loading local model from {model_path} ...")
            ml_models["tokenizer"] = AutoTokenizer.from_pretrained(model_path)
            ml_models["model"] = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            print("Local model not found — falling back to 'distilbert-base-uncased-finetuned-sst-2-english' from Hugging Face Hub.")
            ml_models["tokenizer"] = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            ml_models["model"] = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        ml_models["device"] = device
        # move to device (may be CPU)
        try:
            ml_models["model"].to(device)
        except Exception as e:
            # Log but don't crash — model may be on meta device in some edge cases
            print(f"Warning: moving model to device failed: {e}")

        ml_models["model"].eval()
        print(f"✅ Models loaded successfully on device: {ml_models['device']}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
    yield
    ml_models.clear()
    print("✅ Sentiment service shutdown complete.")

app = FastAPI(lifespan=lifespan)

# -------------------------
# Request / Response models
# -------------------------
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

# -------------------------
# Helper predict function
# -------------------------
def _predict_from_text(text: str):
    if "model" not in ml_models:
        raise RuntimeError("Model not loaded")

    pre = ml_models["preprocessor"]
    tokenizer = ml_models["tokenizer"]
    model = ml_models["model"]
    device = ml_models["device"]

    processed = pre.preprocess_text(text)
    inputs = tokenizer(processed, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])

    # map id->label using model config if available
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        label = id2label.get(pred_id, str(pred_id))
    else:
        label = str(pred_id)

    prob_dict = {id2label.get(i, str(i)): float(p) for i, p in enumerate(probs)} if id2label else {str(i): float(p) for i, p in enumerate(probs)}
    return label, confidence, prob_dict

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
def read_root():
    return {"status": "Sentiment Analysis Service is running."}

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        label, confidence, probs = _predict_from_text(request.text)
        return SentimentResponse(sentiment=label, confidence=confidence, probabilities=probs)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during sentiment analysis.")

# Compatibility endpoint used by some frontends (/predict)
@app.post("/predict")
async def predict_compat(request: SentimentRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        label, confidence, probs = _predict_from_text(request.text)
        # older frontends expect "sentiment" and "score"
        return {"sentiment": label, "score": confidence, "probabilities": probs}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during sentiment analysis.")
