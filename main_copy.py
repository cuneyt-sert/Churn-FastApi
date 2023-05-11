from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle


app = FastAPI()


class modelShema2(BaseModel):
    creditscore:int
    age:int
    tenure:int
    balance:int
    numofproducts:int
    estimatedsalary:int
    


@app.get("/")
def home():
    return {"mesaj": "Logistic Regression Ml model için predict/log_reg kısmına , KNN için predict/knn kısmına gidin"}

@app.post("/predict/log_reg")
def predict_logreg_ml(predict_value:modelShema2):
    filename = "logreg1_model.pkl"
    load_model = pickle.load(open(filename, "rb"))
    
    df = pd.DataFrame(
        [predict_value.dict().values()],
        columns=predict_value.dict().keys()
    )


    predict = load_model.predict(df)
    return {"Predict":int(predict[0])}
    
@app.post("/predict/knn")
def predict_knn_ml(predict_value:modelShema2):
    filename = "knn_model.pkl"
    load_model = pickle.load(open(filename, "rb"))
    
    df2 = pd.DataFrame(
        [predict_value.dict().values()],
        columns=predict_value.dict().keys()
    )


    predict2 = load_model.predict(df2)
    return {"Predict":int(predict2[0])}
