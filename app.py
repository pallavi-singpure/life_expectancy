from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()

# Load model & scaler
model = joblib.load("life_expectancy_model.pkl")
scaler = joblib.load("scaler (2).pkl")

# Mount static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(
        request: Request,
        year: int = Form(...),
        status: int = Form(...),
        adult_mortality: float = Form(...),
        infant_deaths: float = Form(...),
        alcohol: float = Form(...),
        expenditure: float = Form(...),
        hepatitis: float = Form(...),
        measles: float = Form(...),
        bmi: float = Form(...),
        under_five: float = Form(...),
        polio: float = Form(...),
        total_exp: float = Form(...),
        diphtheria: float = Form(...),
        hiv: float = Form(...),
        gdp: float = Form(...),
        population: float = Form(...),
        thin1: float = Form(...),
        thin2: float = Form(...),
        income: float = Form(...),
        schooling: float = Form(...)
):
    data = [[
        year, status, adult_mortality, infant_deaths, alcohol,
        expenditure, hepatitis, measles, bmi, under_five,
        polio, total_exp, diphtheria, hiv, gdp, population,
        thin1, thin2, income, schooling
    ]]

    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": round(prediction, 2)
        }
    )
