from fastapi import FastAPI
from pydantic import BaseModel
from api.interface import multilabel_pipeline

model = multilabel_pipeline()

app = FastAPI()

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    predictions = model.predict([item.text])
    return predictions[0]