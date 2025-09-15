# CreditSage — AI-Powered Loan Approval Prediction

A Streamlit app that predicts the probability of loan approval by combining structured profile data with sentence-transformer embeddings of the applicant's justification text. It also generates actionable guidance using Groq's Llama models.

## Features
- Futuristic dark UI with glassmorphism and neon accents
- Hybrid model: structured features + text embeddings
- Top-line metrics, gauge visualization with centered value, and progress bar
- Sidebar summary and downloadable assessment report
- Groq AI feedback with concise assessment, tips, and a verdict line
- Expanded inputs: granular loan tenure and conditional employment role fields

## Project Structure
```
loan-approval-app/
├── app.py                     # Streamlit app
├── model_training_simple.py   # Simple trainer that saves weights
├── model_training.py          # Full training script (legacy pattern kept)
├── model_weights.pkl          # Trained model weights (loaded by app.py)
├── embedder.pkl               # SentenceTransformer model
├── scaler.pk1                 # Scaler for structured inputs
├── data/
│   └── loan_data.csv          # Training dataset (headers may contain leading spaces)
├── .streamlit/
│   └── config.toml            # Theme config (dark/neon)
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.12
- Windows PowerShell or similar shell

## Quick Start
1. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies
```powershell
pip install -r requirements.txt
```
3. Run the app
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
4. Open the local URL shown in the terminal (e.g., http://localhost:8501)

## Model Artifacts
We avoid Keras deserialization issues by rebuilding the model architecture in `app.py` and loading weights from `model_weights.pkl`.
- `embedder.pkl` — sentence-transformer used to encode justification text
- `scaler.pk1` — scikit-learn scaler for numeric features
- `model_weights.pkl` — saved via `joblib.dump(model.get_weights(), ...)`

If you need to retrain:
```powershell
.\.venv\Scripts\python.exe model_training_simple.py
```
This will regenerate `model_weights.pkl` (and ensure `embedder.pkl` / `scaler.pk1` exist).

## Groq AI Feedback
The app uses Groq's `llama-3.1-8b-instant` to produce guidance. The API key is configured server-side in `app.py` to avoid any UI exposure. For production, prefer environment variables or `st.secrets` instead of hardcoding secrets.

## Configuration and Theming
We ship a dark neon theme via `.streamlit/config.toml`. You can further tweak colors and fonts there. The app also injects some scoped CSS for glass panels and neon accents.

## Deployment Notes
- Hide secrets: move API keys to environment variables or Streamlit secrets (`.streamlit/secrets.toml`) in production.
- Pin dependencies via `requirements.txt` and build within a clean virtual environment.
- Ensure the three runtime artifacts are present on the server: `model_weights.pkl`, `embedder.pkl`, `scaler.pk1`.

## Troubleshooting
- "Could not deserialize class 'Functional'": use the shipped approach (weights + rebuild) instead of loading `.keras` files.
- Missing packages: re-run `pip install -r requirements.txt` in the active venv.
- Import errors with Keras/Transformers: ensure `tf-keras` is installed and used as in `app.py`.
- Streamlit doesn’t start: check the terminal for errors and confirm you’re using the venv’s `streamlit.exe`.

## License
This project is provided as-is for demo and educational purposes.
