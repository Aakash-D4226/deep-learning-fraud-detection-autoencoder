# ============================================================
# DEEPGUARD
# FRAUD PREDICTION
# ============================================================

import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model(
    "models/fraud_autoencoder.keras"
)


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(
    "models/scaler.pkl"
)


# ============================================================
# LOAD THRESHOLD
# ============================================================

threshold = joblib.load(
    "models/threshold.pkl"
)


print(
    "Loaded anomaly threshold:",
    threshold
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def detect_fraud(transaction):

    transaction = np.array(
        transaction,
        dtype=float
    ).reshape(1, -1)


    # Scale transaction
    transaction_scaled = scaler.transform(
        transaction
    )


    # Reconstruction
    reconstruction = model.predict(
        transaction_scaled,
        verbose=0
    )


    # Reconstruction error
    error = np.mean(
        np.square(
            transaction_scaled -
            reconstruction
        )
    )


    # Fraud decision
    if error > threshold:

        result = "FRAUD"

    else:

        result = "NORMAL"


    return result, error


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/creditcard.csv"
)


# ============================================================
# SELECT ONE TRANSACTION
# ============================================================

sample = df.drop(
    "Class",
    axis=1
).drop(
    "Time",
    axis=1
).iloc[0].values


# ============================================================
# RUN PREDICTION
# ============================================================

result, error = detect_fraud(
    sample
)


# ============================================================
# DISPLAY
# ============================================================

print(
    "\n================================"
)

print(
    "TRANSACTION ANALYSIS"
)

print(
    "================================"
)

print(
    "Reconstruction Error:",
    error
)

print(
    "Threshold:",
    threshold
)

print(
    "Prediction:",
    result
)

print(
    "================================"
)