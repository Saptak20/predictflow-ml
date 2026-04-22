# PredictFlow ML (Streamlit)

A minimal, deployable ML web app built with Streamlit.

## Project Structure

- `app.py` - Streamlit UI + model inference
- `model.pkl` - Trained ML model file (dummy model generated for now)
- `train_dummy_model.py` - Script to generate the dummy model
- `requirements.txt` - Python dependencies

## Data Flow

1. User enters values in the Streamlit UI.
2. Input is validated and converted to numeric format.
3. `model.pkl` is loaded using `joblib`.
4. Model generates prediction.
5. Prediction is shown in the UI.

## Local Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Generate Dummy Model (if needed)

```bash
python train_dummy_model.py
```

## GitHub Push (Exact repo)

```bash
git init
git add .
git commit -m "Initial Streamlit ML app"
git branch -M main
git remote add origin https://github.com/Saptak20/predictflow-ml.git
git push -u origin main
```

## Streamlit Cloud Deployment

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app**.
3. Select repository: `Saptak20/predictflow-ml`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **Deploy**.

## Common Fixes

- Missing model: Run `python train_dummy_model.py` or upload your own `model.pkl`.
- Import errors: Run `python -m pip install -r requirements.txt`.
- Dependency mismatch on cloud: Ensure `requirements.txt` is committed and pushed.
