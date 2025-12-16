import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from collections import Counter
from pydantic import BaseModel
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import contextlib
import re
import unicodedata
from underthesea import word_tokenize
import os

STOPWORDS_FILE = 'vietnamese-stopwords.txt'
MODEL_PATH = 'vinai/phobert-base'
MLP_MODEL_PATH = 'models/news_classifier_mlp.pkl' 
LE_MODEL_PATH = 'models/label_encoder.pkl'

class NewsInput(BaseModel):
    title: str
    content: str

ml_models = {}
STOPWORDS = set()

def load_stopwords_global():
    global STOPWORDS
    try:
        if os.path.exists(STOPWORDS_FILE):
            with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        STOPWORDS.add(word)
                        STOPWORDS.add(word.replace(' ', '_'))
    except Exception:
        pass

def text_preprocess(text):
    if not isinstance(text, str) or text is None:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S*@\S*\s?', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        text = word_tokenize(text, format='text')
    except:
        pass
    if text and STOPWORDS:
        words = text.split()
        clean_words = [w for w in words if w not in STOPWORDS]
        text = " ".join(clean_words)
    return text

def get_embeddings(text_list, max_len=256):
    tokenizer = ml_models['tokenizer']
    bert = ml_models['bert']
    device = ml_models['device']
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    load_stopwords_global()
    try:
        if os.path.exists(MLP_MODEL_PATH) and os.path.exists(LE_MODEL_PATH):
            ml_models['le'] = joblib.load(LE_MODEL_PATH)
            ml_models['mlp'] = joblib.load(MLP_MODEL_PATH)
        
        ml_models['tokenizer'] = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        ml_models['bert'] = AutoModel.from_pretrained(MODEL_PATH)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ml_models['bert'].to(device)
        ml_models['device'] = device
    except Exception as e:
        print(f"Error loading models: {e}")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

app.mount("/styles", StaticFiles(directory="templates/styles"), name="styles")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: NewsInput):
    try:
        clean_title = text_preprocess(data.title)
        clean_content = text_preprocess(data.content) 
        full_text = clean_title + " " + clean_content
        words = full_text.split()
        
        most_common = Counter(words).most_common(8)
        
        freq_chart_data = [{"word": word.replace('_', ' '), "count": count} for word, count in most_common]
        raw_keywords = [word for word, count in most_common]
        display_keywords = [d['word'] for d in freq_chart_data]

        top5_kw = [w for w, c in most_common[:5]]
        top5_display = [w.replace('_', ' ') for w in top5_kw]
        raw_sentences = re.split(r'[.!?]+', data.content.lower())
        matrix = []
        
        for w1 in top5_kw:
            row = []
            for w2 in top5_kw:
                count = 0
                for sent in raw_sentences:
                    if w1.replace('_', ' ') in sent and w2.replace('_', ' ') in sent:
                        count += 1
                row.append(count)
            matrix.append(row)

        token_count = len(clean_content.split())
        title_emb = get_embeddings([clean_title], max_len=64)
        content_emb = get_embeddings([clean_content], max_len=256)
        token_feat = np.array([[token_count]])
        full_features = np.hstack([title_emb, content_emb, token_feat])
        
        mlp = ml_models['mlp']
        le = ml_models['le']
        probs = mlp.predict_proba(full_features)[0]
        
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_labels = le.inverse_transform(top3_idx)
        top3_probs = probs[top3_idx]
        prob_chart_data = [{"label": l, "prob": round(p * 100, 2)} for l, p in zip(top3_labels, top3_probs)]

        sentences = re.split(r'[.!?]+', data.content)
        sentences = [s for s in sentences if s.strip()]
        avg_sent_len = round(token_count / len(sentences)) if sentences else 0

        return {
            "status": "success",
            "prediction": top3_labels[0],
            "confidence": f"{top3_probs[0]*100:.2f}%",
            "keywords": raw_keywords,
            "display_keywords": display_keywords, 
            "prob_chart_data": prob_chart_data, 
            "freq_chart_data": freq_chart_data,
            "sentence_count": len(sentences),
            "avg_sent_len": avg_sent_len,
            "heatmap_labels": top5_display,
            "heatmap_matrix": matrix 
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)