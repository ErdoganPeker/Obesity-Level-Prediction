from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Obesity Level Prediction")
templates = Jinja2Templates(directory="templates")

model = None
le_gender = LabelEncoder()
le_activity = LabelEncoder()
le_target = LabelEncoder()

OBESITY_CLASSES = [
    "Insufficient Weight",
    "Normal Weight",
    "Overweight Level I",
    "Overweight Level II",
    "Obesity Type I",
    "Obesity Type II",
    "Obesity Type III",
]


@app.on_event("startup")
def train():
    global model, le_gender, le_activity, le_target
    np.random.seed(42)
    n = 3000
    genders = np.random.choice(["Male", "Female"], n)
    ages = np.random.randint(15, 65, n)
    heights = np.random.randint(150, 195, n).astype(float)
    weights = np.random.randint(45, 140, n).astype(float)
    activity = np.random.choice(["Low", "Medium", "High"], n)
    veg_freq = np.random.randint(1, 4, n)

    bmi = weights / ((heights / 100) ** 2)

    labels = []
    for b in bmi:
        if b < 18.5:
            labels.append("Insufficient Weight")
        elif b < 25:
            labels.append("Normal Weight")
        elif b < 27.5:
            labels.append("Overweight Level I")
        elif b < 30:
            labels.append("Overweight Level II")
        elif b < 35:
            labels.append("Obesity Type I")
        elif b < 40:
            labels.append("Obesity Type II")
        else:
            labels.append("Obesity Type III")

    gender_enc = le_gender.fit_transform(genders)
    activity_enc = le_activity.fit_transform(activity)
    labels_enc = le_target.fit_transform(labels)

    X = np.column_stack([gender_enc, ages, heights, weights, activity_enc, veg_freq])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, labels_enc)


class PredictRequest(BaseModel):
    gender: str
    age: int
    height: float
    weight: float
    activity: str
    veg_freq: int = 2


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: PredictRequest):
    gender_enc = le_gender.transform([data.gender])[0]
    activity_enc = le_activity.transform([data.activity])[0]
    X = np.array([[gender_enc, data.age, data.height, data.weight, activity_enc, data.veg_freq]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = float(proba[pred])
    label = le_target.inverse_transform([pred])[0]
    bmi = data.weight / ((data.height / 100) ** 2)
    return {"prediction": label, "confidence": round(confidence, 4), "bmi": round(bmi, 1)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
