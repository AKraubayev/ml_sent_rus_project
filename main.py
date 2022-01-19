from fastapi import FastAPI
import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello UrFU"}

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True)

@torch.no_grad()

def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    if predicted == 0:
      print("NEUTRAL")
    elif predicted == 1:
      print("POSITIVE")
    else:
      print("NEGATIVE")
    return predicted

text = input("Введите текст: ")

predict(text)

@app.post("/predict/")

def predict(item: text):
    return predicted(item.text)[0]
