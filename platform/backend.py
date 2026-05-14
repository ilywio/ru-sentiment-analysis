import logging
import torch
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

from database import Prediction, SessionLocal, init_db

logging.basicConfig(level=logging.INFO)
logging.info("Starting FastAPI application")

MODEL_PATH = "../models/my_model"

ID2TEXT = {
    0: "негативная",
    1: "позитивная",
    2: "мусор",
    3: "нейтральная"
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


def load_model():
    logging.info("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logging.info("Model loaded successfully")

    return tokenizer, model, device


tokenizer, model, device = load_model()


class TextRequest(BaseModel):
    text: str


@app.post("/predict/")
async def predict(request: TextRequest):
    logging.info(f"Received text: {request.text}")

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    predicted_class_text = ID2TEXT[pred]

    logging.info(f"Predicted class: {predicted_class_text}")

    db = SessionLocal()
    try:
        db_obj = Prediction(
            comment=request.text,
            predicted_class=predicted_class_text
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
    except SQLAlchemyError as e:
        db.rollback()
        logging.error(f"DB error: {e}")
    finally:
        db.close()

    return {"predicted_class": predicted_class_text}


@app.get("/")
async def root():
    return {"message": "Emotion Classification API (BERT)"}


@app.get("/hello")
async def hello():
    return {"message": "Hello, world!"}


@app.get("/v1/hello")
async def hello_v1():
    return {"message": "Hello from v1!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
