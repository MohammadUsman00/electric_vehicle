PROJECT TITLE : AI-Powered Range Estimation for Electric Vehicles Using Battery Data
💡 Problem Statement

With the increasing adoption of Electric Vehicles (EVs), predicting battery performance and remaining driving range has become critical. Battery degradation affects the range, cost, and reliability of EVs. The goal of this project is to develop a machine learning model that accurately predicts the battery health and remaining driving range of an EV using operational data such as charge cycles, temperature, voltage, and usage patterns.
🎯 Objectives

1. Collect and preprocess EV battery data from Kaggle.


2. Perform data visualization to identify key factors affecting battery life.


3. Build regression models (Linear Regression, Random Forest, XGBoost) to predict range or health.


4. Evaluate model accuracy using metrics (R², RMSE, MAE).


5. Visualize the predicted vs actual values and feature importance

## How to run

1. Create and activate a Python virtual environment (recommended).
2. Install the project dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) Set your OpenAI API key if you want to enable the chat assistant:

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

4. Run the Streamlit app:

```powershell
streamlit run app.py
```

Notes:
- The app will allow you to upload a CSV, train a regression model, save the trained model, and make predictions. If XGBoost is not installed the app will still run with Linear Regression and Random Forest.
- The example dataset `Experimental_data_aged_cell.csv` can be used as a starting point for exploration.

If you have a Gemini key (or prefer a separate environment variable name) you can set
`GEMINI_API_KEY` instead. The app prefers `GEMINI_API_KEY` if present, otherwise it will use
`OPENAI_API_KEY`.

Example (PowerShell):

```powershell
setx GEMINI_API_KEY "your_gemini_api_key_here"
# Restart your shell or sign out/in so the new environment variable is available to GUI apps
```

Important: never commit API keys to the repository. Use environment variables or a secret manager.