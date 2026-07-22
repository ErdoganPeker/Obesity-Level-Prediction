from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn, os
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Obesity Level Prediction")
templates = Jinja2Templates(directory="templates")
_dir = os.path.dirname(os.path.abspath(__file__))

model = None
model_accuracy = None
feature_importance = []

# All categorical columns in the dataset that need label-encoding, plus the
# target. Height/Age/Weight/FCVC/NCP/CH2O/FAF/TUE are already numeric.
CATEGORICAL_COLS = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]

FEATURE_COLS = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "family_history_with_overweight",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS",
]

encoders = {col: LabelEncoder() for col in CATEGORICAL_COLS}
le_target = LabelEncoder()


@app.on_event("startup")
def train():
    global model, le_target, model_accuracy, feature_importance
    df = pd.read_csv(os.path.join(_dir, "ObesityDataSet_raw_and_data_sinthetic.csv"))

    df_enc = df.copy()
    for col, enc in encoders.items():
        df_enc[col] = enc.fit_transform(df_enc[col])

    y = le_target.fit_transform(df["NObeyesdad"])
    X = df_enc[FEATURE_COLS].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    model_accuracy = round(float(accuracy_score(y_test, y_pred)), 4)

    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )
    feature_importance = [
        {"feature": name, "importance": round(float(score), 4)}
        for name, score in importances
    ]


class PredictRequest(BaseModel):
    gender: str
    age: float
    height: float  # meters, e.g. 1.70
    weight: float
    family_history: str  # "yes"/"no"
    favc: str  # "yes"/"no"
    fcvc: float  # 1-3
    ncp: float  # 1-4
    caec: str  # no/Sometimes/Frequently/Always
    smoke: str  # "yes"/"no"
    ch2o: float  # 1-3
    scc: str  # "yes"/"no"
    faf: float  # 0-3
    tue: float  # 0-2
    calc: str  # no/Sometimes/Frequently/Always
    mtrans: str  # transportation mode


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: PredictRequest):
    row = {
        "Gender": encoders["Gender"].transform([data.gender])[0],
        "Age": data.age,
        "Height": data.height,
        "Weight": data.weight,
        "family_history_with_overweight": encoders["family_history_with_overweight"].transform([data.family_history])[0],
        "FAVC": encoders["FAVC"].transform([data.favc])[0],
        "FCVC": data.fcvc,
        "NCP": data.ncp,
        "CAEC": encoders["CAEC"].transform([data.caec])[0],
        "SMOKE": encoders["SMOKE"].transform([data.smoke])[0],
        "CH2O": data.ch2o,
        "SCC": encoders["SCC"].transform([data.scc])[0],
        "FAF": data.faf,
        "TUE": data.tue,
        "CALC": encoders["CALC"].transform([data.calc])[0],
        "MTRANS": encoders["MTRANS"].transform([data.mtrans])[0],
    }
    X = pd.DataFrame([row])[FEATURE_COLS].astype(float)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = float(proba[pred])
    label = le_target.inverse_transform([pred])[0].replace("_", " ")

    class_probabilities = sorted(
        (
            {"label": cls.replace("_", " "), "probability": round(float(p), 4)}
            for cls, p in zip(le_target.classes_, proba)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )

    bmi = data.weight / (data.height ** 2)
    bmi = round(bmi, 1)

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "bmi": bmi,
        "bmi_category": _bmi_category(bmi),
        "class_probabilities": class_probabilities,
        "feature_importance": feature_importance,
        "model_accuracy": model_accuracy,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
