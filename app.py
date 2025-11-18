import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel  , computed_field , Field
from typing import Annotated  , Literal
import pickle

with open ("churn_model.pkl","rb") as f :
    model=pickle.load(f)


app= FastAPI()


class user_input(BaseModel):
    CreditScore : Annotated[ float , Field(..., description="Credit card score of the user")]   
    Geography  : Annotated[ str , Field(..., description="Area where user lives")]    
    Gender    : Annotated[ str,Literal['Male' , 'Female'], Field(..., description="Gender of the user")]     
    Age      : Annotated[ int , Field(...,gt=0, description="Age of the user")]      
    Tenure   : Annotated[ int , Field(..., description="Tenure ")]      
    Balance  : Annotated[ float, Field(..., description="Balance of the user")]      
    NumOfProducts : Annotated[ int , Field(..., description="Products number owned by the user")] 
    HasCrCard     : Annotated[ int , Field(..., description="how many cars user has")] 
    IsActiveMember : Annotated[ int , Field(..., description="active card numbers")]
    EstimatedSalary: Annotated[ float , Field(..., description="Est Salary of the user")]



@app.get("/")
def hello():
    return {'message':"Hello to the Churn Modeling Web Interface "}


@app.get("/about")
def about():
    return {"message":"Checking the eligibity for the Churn modeling"}


@app.post("/predict")
def predict_churn(data: user_input):

    # Convert pydantic model to dict
    payload = data.dict()

    # Create initial dataframe
    df = pd.DataFrame([payload])

    # ---------------------------------------
    # MANUAL ONE-HOT ENCODING (MUST MATCH TRAINING)
    # ---------------------------------------

    # Geography dummy columns
    df["Geography_France"] = 1 if data.Geography == "France" else 0
    df["Geography_Spain"] = 1 if data.Geography == "Spain" else 0
    df["Geography_Germany"] = 1 if data.Geography == "Germany" else 0

    # Gender dummy columns
    df["Gender_Male"] = 1 if data.Gender == "Male" else 0
    df["Gender_Female"] = 1 if data.Gender == "Female" else 0

    # ---------------------------------------
    # DROP RAW COLUMNS (because your model didn't train on them)
    # ---------------------------------------
    df.drop(columns=["Geography", "Gender"], inplace=True)

    # ---------------------------------------
    # ENSURE COLUMN ORDER (CRITICAL)
    # ---------------------------------------
    expected_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Geography_France",
        "Geography_Spain",
        "Geography_Germany",
        "Gender_Female",
        "Gender_Male",
    ]

    df = df[expected_cols]   # reorder

    # ---------------------------------------
    # Predict
    # ---------------------------------------
    prediction = model.predict(df)[0]

    return {"predicted_category": int(prediction)}








