# 🧠 Bankruptcy Prediction App (Final Project)

This is a full-stack machine learning application that predicts the risk of company bankruptcy using financial indicators.

It includes:

- ✅ A **FastAPI** backend for predictions
- ✅ A **Streamlit** frontend for user interaction
- ✅ A **Logistic Regression** model trained on 95 financial features
- ✅ **SHAP Beeswarm plots** for model interpretability
- ✅ **Great Expectations** for data validation
- ✅ **Dockerized deployment** using Docker Compose
- ✅ An exported OpenAPI **data contract**



I couldn't use streamlit because it was giving me errors, i tried fixing it but the result was same it was giving me error, so i used swagger
---

## 📁 Project Structure


---

## 🚀 How to Run (Locally with Docker)

1. **Clone the repository**:
```bash
git clone https://github.com/aseemsyed10/Bankruptcy_App_Final.git
cd Bankruptcy_App_Final
Build the Docker containers:
docker compose build
Run the app:
docker compose up
{
  "features": [0.1, 0.3, ..., 0.97]  // Total of 95 values
}
OpenAPI schema is available at:
http://localhost:8000/openapi.json
Also included in this repo as openapi_schema.json
