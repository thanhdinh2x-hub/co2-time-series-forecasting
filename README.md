# CO2 Time Series Forecasting

A machine learning project for forecasting CO2 concentrations using historical time-series data and Linear Regression.

## Features

- Missing value interpolation
- Lag feature generation
- Time-series train/test split
- Linear Regression model
- MAE, MSE and R2 evaluation
- Visualization of predictions
- Model serialization using Pickle

## Project Structure

```text
co2-time-series-forecasting/
├── data/
│   └── co2.csv
├── models/
│   └── co2_model.pkl
├── src/
│   └── train_model.py
├── README.md
├── requirements.txt
└── .gitignore