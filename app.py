import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tf_keras import layers
from tf_keras.models import Model
import joblib
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from groq import Groq
from io import StringIO
import os

# Config
st.set_page_config(page_title="CreditSage • Loan Approval AI", page_icon="🏦", layout="wide")

# Futuristic styling
st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(1200px 600px at 10% 10%, #0b1530 0%, #050510 60%) no-repeat fixed;
            }
            /* Glass panels */
            .glass {
                background: rgba(10, 16, 32, 0.7);
                border: 1px solid rgba(0, 191, 255, 0.15);
                border-radius: 16px;
                box-shadow: 0 0 30px rgba(0, 191, 255, 0.08);
                padding: 1.2rem 1.2rem 0.6rem 1.2rem;
                backdrop-filter: blur(8px);
            }
            /* Neon accents */
            .neon { color: #00BFFF; }
            .subtle { color: #9fb3ff; }
            /* Buttons */
            div.stButton > button:first-child {
                background: linear-gradient(90deg, #0066ff 0%, #00bfff 100%);
                color: #ffffff; border: 0; border-radius: 12px; padding: 0.6rem 1rem;
                box-shadow: 0 10px 20px rgba(0, 102, 255, 0.25);
            }
            div.stButton > button:first-child:hover { filter: brightness(1.1); }
            /* Cards spacing */
            .block-container { padding-top: 2rem; padding-bottom: 3rem; }
            /* Gauge container */
            .result-card { border-top: 1px solid rgba(0,191,255,0.2); margin-top: .5rem; padding-top: .75rem; }
            /* Markdown sections */
            h1, h2, h3 { color: #E0E7FF; }
            .small { font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
)

# Load model & scaler
scaler = joblib.load("scaler.pk1")
embedder = joblib.load("embedder.pkl")
model_weights = joblib.load("model_weights.pkl")

# Rebuild the model architecture
def create_model():
    struct_inp = layers.Input(shape=(3,), name="struct_input")
    x1 = layers.Dense(32, activation="relu")(struct_inp)
    
    text_inp = layers.Input(shape=(384,), name="text_input")
    x2 = layers.Dense(64, activation="relu")(text_inp)
    
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=[struct_inp, text_inp], outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Set the weights from our trained model
    model.set_weights(model_weights)
    return model

model = create_model()

# Groq setup (server-side; not exposed in UI)
# Reads from Streamlit secrets or environment variable.
client = None
_secrets_key = None
try:
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        _secrets_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

api_key = _secrets_key or os.environ.get("GROQ_API_KEY")
if api_key:
    try:
        client = Groq(api_key=api_key)
    except Exception:
        client = None

# UI
st.markdown(
    """
    <div class="glass">
      <h1>🏦 CreditSage <span class="neon">AI</span></h1>
      <p class="subtle small">Predict loan approval chances with a blended model that understands your profile and justification text.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        # Inputs - money & credit
        currency = st.selectbox(
            "Currency",
            ["USD", "INR", "EUR", "GBP", "JPY", "CAD", "AUD", "SGD", "AED", "ZAR"],
            help="Select the currency used for income and loan amount.",
        )
        income = st.number_input(
            f"Annual Income ({currency})",
            min_value=0,
            step=1000,
            help="Your gross annual income before taxes.",
        )
        loan_amt = st.number_input(
            f"Loan Amount ({currency})",
            min_value=0,
            step=1000,
            help="The amount you want to borrow.",
        )
        credit_score = st.slider(
            "Credit Score",
            300,
            850,
            650,
            help="Approximate credit score on a 300–850 scale.",
        )

        # Tenure options (more granular)
        tenure_months = [3, 6, 9, 12, 15, 18, 24, 30, 36, 48, 60, 72, 84, 96, 120, 180, 240, 300, 360]
        tenure_labels = [f"{m} months" if m < 12 or m % 12 != 0 else f"{m//12} years" for m in tenure_months]
        tenure_idx = st.selectbox(
            "Loan Tenure",
            options=list(range(len(tenure_months))),
            format_func=lambda i: tenure_labels[i],
            help="Select a payback period that fits your budget.",
        )
        tenure = tenure_labels[tenure_idx]
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        # Employment
        employment = st.selectbox(
            "Employment Type",
            [
                "Salaried",
                "Self-Employed",
                "Freelancer/Contractor",
                "Business Owner",
                "Unemployed",
                "Student",
                "Retired",
            ],
            help="Select the category that best describes your work status.",
        )
        role = ""
        if employment in ("Salaried", "Freelancer/Contractor"):
            role = st.text_input("Job Title / Role", placeholder="e.g., Software Engineer, Data Analyst")
        elif employment in ("Self-Employed", "Business Owner"):
            role = st.text_input("Business Type / Profession", placeholder="e.g., Retail, Consulting, Bakery")

        # Justification
        justification = st.text_area(
            "Loan Justification",
            placeholder="Explain why you need the loan and how you'll repay it...",
            help="A clear, concise justification improves AI guidance.",
            height=150,
        )

        # Quick derived metrics in-panel
        dti = round(float(loan_amt) / float(income) * 100, 2) if income else None
        if dti is not None:
            st.caption(f"Estimated Debt-to-Income (loan-only): {dti}%")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("\n")
run = st.button("🔮 Check Loan Approval")
if run:
    if justification.strip() == "":
        st.warning("⚠️ Please provide loan justification.")
    else:
        # Prepare structured features
        struct_input = pd.DataFrame([[income, loan_amt, 1 if credit_score > 650 else 0]],
                                    columns=["ApplicantIncome", "LoanAmount", "Credit_History"])
        struct_scaled = scaler.transform(struct_input)

        # Text embedding
        text_emb = embedder.encode([justification])

        # Prediction
        score = model.predict([struct_scaled, text_emb])[0][0]
        pct = round(float(score) * 100, 2)

        # Circle meter
        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Approval Chance", f"{pct}%")
        m2.metric("Credit Score", f"{credit_score}")
        m3.metric("Income", f"{income:,} {currency}")
        m4.metric("Loan", f"{loan_amt:,} {currency}")

        # Circle meter with centered number
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=pct,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00BFFF" if pct >= 50 else "#ff4d4f"},
                'bgcolor': 'rgba(0,0,0,0)'
            },
            title={'text': "Approval Chance (%)"}
        ))
        # Place number in the middle of the gauge area
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"{pct}",
            showarrow=False,
            font=dict(size=48, color="#E0E7FF"),
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.markdown("<div class='glass result-card'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.progress(int(pct))
        st.markdown("</div>", unsafe_allow_html=True)

        # Groq summary
        outlook = "positive" if pct >= 50 else "negative"
        profile_block = f"""
        Currency: {currency}
        Annual Income: {income}
        Loan Amount: {loan_amt}
        Credit Score: {credit_score}
        Loan Tenure: {tenure}
        Employment Type: {employment}{' (' + role + ')' if role else ''}
        Estimated DTI (loan-only): {str(dti) + '%' if dti is not None else 'n/a'}
        """
        prompt = f"""
You are CreditSage, an expert loan advisor. Analyze the applicant profile below and respond with:
1) A 3-4 sentence assessment in a professional, empathetic tone.
2) Three numbered, actionable suggestions to improve approval odds.
3) A one-line summary starting with 'Verdict:' (Approved likely / Borderline / Unlikely) based on a {pct}% model estimate and {outlook} outlook.

Applicant Profile:\n{profile_block}\n\nJustification:\n{justification}
Keep the total under 160 words. Avoid repeating the numbers verbatim; focus on insight.
"""

        response_text = None
        if client is not None:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                )
                response_text = chat_completion.choices[0].message.content
            except Exception as e:
                st.sidebar.error(f"Error accessing Groq API: {e}")
        if response_text is None:
            response_text = f"AI feedback unavailable. Outlook appears {outlook}. Focus on improving credit profile and ensuring repayment capacity."

        st.markdown("###  AI Feedback")
        st.markdown(f"<div class='glass'><p>{response_text}</p></div>", unsafe_allow_html=True)

        # Sidebar summary
        with st.sidebar:
            st.markdown("---")
            st.subheader("Your Summary")
            st.caption(f"Currency: {currency}")
            st.caption(f"Income: {income:,} {currency}")
            st.caption(f"Loan: {loan_amt:,} {currency}")
            st.caption(f"Credit: {credit_score}")
            st.caption(f"Tenure: {tenure}")
            st.caption(f"Employment: {employment}{' — ' + role if role else ''}")
            if dti is not None:
                st.caption(f"Est. DTI: {dti}%")

        # Download report
        report = StringIO()
        report.write("CreditSage Loan Assessment\n")
        report.write("\n---\n")
        report.write(f"Approval chance: {pct}% ({outlook})\n")
        report.write(f"Income: {income:,} {currency}\n")
        report.write(f"Loan Amount: {loan_amt:,} {currency}\n")
        report.write(f"Credit Score: {credit_score}\n")
        report.write(f"Tenure: {tenure}\n")
        report.write(f"Employment: {employment}{' - ' + role if role else ''}\n")
        if dti is not None:
            report.write(f"Estimated DTI: {dti}%\n")
        report.write("\nAI Feedback:\n")
        report.write(response_text)
        st.download_button(
            label="⬇️ Download Assessment",
            data=report.getvalue().encode("utf-8"),
            file_name="creditsage_assessment.txt",
            mime="text/plain",
        )

        # Disclaimer
        st.caption("This tool provides indicative guidance only and is not financial advice.")
