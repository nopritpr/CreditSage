# CreditSage — AI-Powered Loan Approval Prediction

A Streamlit app that predicts the probability of loan approval by combining structured profile data with sentence-transformer embeddings of the applicant's justification text. It also generates actionable guidance using Groq's Llama models.
<img width="1611" height="741" alt="Screenshot 2025-09-16 104932" src="https://github.com/user-attachments/assets/fa94f280-e5ac-427f-a90b-e2c660d5b5a6" />
<img width="1634" height="717" alt="Screenshot 2025-09-16 104953" src="https://github.com/user-attachments/assets/072a4d9e-5795-4d14-8415-c20676b97d3c" />
<img width="1672" height="502" alt="Screenshot 2025-09-16 105003" src="https://github.com/user-attachments/assets/bb1808b4-33af-4afd-8ca2-72b7ce557955" />


<img width="1545" height="837" alt="Screenshot 2025-09-16 105319" src="https://github.com/user-attachments/assets/3c04800d-0078-4f18-b1eb-145a9fc57d42" />
<img width="1605" height="890" alt="Screenshot 2025-09-16 105337" src="https://github.com/user-attachments/assets/3f90cd29-ce75-4272-9db0-c0cec6b7c71c" />



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
3. Configure your Groq API key (local)
	- Create a `.env` file (or copy `.env.example` to `.env`) and set:
```env
GROQ_API_KEY=sk_your_key_here
```
	- Alternatively, set an environment variable in your shell/session:
```powershell
$env:GROQ_API_KEY = "sk_your_key_here"
```
	- In Streamlit Cloud or production, use `st.secrets` instead.

4. Run the app
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
5. Open the local URL shown in the terminal (e.g., http://localhost:8501)

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
The app uses Groq's `llama-3.1-8b-instant` to produce guidance. The API key is loaded in this order: `st.secrets` → OS environment variables (including those from `.env`). Keys are never requested in the UI. For production, prefer `st.secrets` or infrastructure-level env vars.

## Configuration and Theming
We ship a dark neon theme via `.streamlit/config.toml`. You can further tweak colors and fonts there. The app also injects some scoped CSS for glass panels and neon accents.

## Deployment Notes
- Hide secrets: move API keys to environment variables or Streamlit secrets (`.streamlit/secrets.toml`) in production.
- Pin dependencies via `requirements.txt` and build within a clean virtual environment.
- Ensure the three runtime artifacts are present on the server: `model_weights.pkl`, `embedder.pkl`, `scaler.pk1`.

## Troubleshooting
- "Could not deserialize class 'Functional'": use the shipped approach (weights + rebuild) instead of loading `.keras` files.
- Missing packages: re-run `pip install -r requirements.txt` in the active venv.
- Import errors with Keras/Transformers: ensure `tf-keras` is installed and imported as in `app.py` (`from tf_keras import layers`).
- Streamlit doesn’t start: check the terminal for errors and confirm you’re using the venv’s `streamlit.exe`.

## License
This project is provided as-is for demo and educational purposes.
