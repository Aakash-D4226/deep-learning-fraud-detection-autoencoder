# ============================================================
# DEEPGUARD
# Deep Learning Credit Card Fraud Detection
# Using Unsupervised Autoencoder
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    "data/creditcard.csv"
)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 4. BASIC DATA CHECK
# ============================================================

print("\nMissing values:")

print(
    df.isnull().sum().sum()
)

print("\nClass distribution:")

print(
    df["Class"].value_counts()
)

print("\nClass percentage:")

print(
    df["Class"]
    .value_counts(normalize=True)
    .mul(100)
)


# ============================================================
# 5. CLASS DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(7, 5))

df["Class"].value_counts().plot(
    kind="bar"
)

plt.title(
    "Credit Card Transaction Distribution"
)

plt.xlabel(
    "Class"
)

plt.ylabel(
    "Number of Transactions"
)

plt.xticks(
    [0, 1],
    ["Normal", "Fraud"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "results/class_distribution.png"
)

plt.close()


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]


# ============================================================
# 7. REMOVE TIME
# ============================================================

X = X.drop(
    "Time",
    axis=1
)


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 9. TRAIN ONLY ON NORMAL TRANSACTIONS
# ============================================================

X_train_normal = X_train[
    y_train == 0
]

print(
    "\nNormal transactions used for training:",
    len(X_train_normal)
)


# ============================================================
# 10. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_normal_scaled = scaler.fit_transform(
    X_train_normal
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 11. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print(
    "\nScaler saved."
)


# ============================================================
# 12. BUILD AUTOENCODER
# ============================================================

input_dim = X_train_normal_scaled.shape[1]

print(
    "\nInput features:",
    input_dim
)


# Encoder
inputs = Input(
    shape=(input_dim,),
    name="transaction_input"
)

x = Dense(
    32,
    activation="relu",
    name="encoder_32"
)(inputs)

x = Dropout(
    0.10
)(x)

x = Dense(
    16,
    activation="relu",
    name="encoder_16"
)(x)

latent = Dense(
    8,
    activation="relu",
    name="latent_space"
)(x)


# Decoder
x = Dense(
    16,
    activation="relu",
    name="decoder_16"
)(latent)

x = Dense(
    32,
    activation="relu",
    name="decoder_32"
)(x)

outputs = Dense(
    input_dim,
    activation="linear",
    name="reconstruction"
)(x)


# Create model
autoencoder = Model(
    inputs,
    outputs,
    name="DeepGuard_Autoencoder"
)


# ============================================================
# 13. COMPILE
# ============================================================

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)


print("\nModel Architecture:")

autoencoder.summary()


# ============================================================
# 14. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ============================================================
# 15. TRAIN AUTOENCODER
# ============================================================

print("\nTraining Autoencoder...")

history = autoencoder.fit(
    X_train_normal_scaled,
    X_train_normal_scaled,

    epochs=20,

    batch_size=256,

    validation_split=0.20,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# 16. SAVE MODEL
# ============================================================

autoencoder.save(
    "models/fraud_autoencoder.keras"
)

print(
    "\nModel saved."
)


# ============================================================
# 17. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Autoencoder Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/training_loss.png"
)

plt.close()


# ============================================================
# 18. RECONSTRUCT NORMAL TRAINING DATA
# ============================================================

print(
    "\nCalculating reconstruction errors..."
)

train_predictions = autoencoder.predict(
    X_train_normal_scaled,
    batch_size=512,
    verbose=0
)


# ============================================================
# 19. TRAINING RECONSTRUCTION ERROR
# ============================================================

train_errors = np.mean(
    np.square(
        X_train_normal_scaled -
        train_predictions
    ),
    axis=1
)

print(
    "\nTraining reconstruction error statistics:"
)

print(
    pd.Series(train_errors).describe()
)


# ============================================================
# 20. INITIAL ANOMALY THRESHOLD
# ============================================================

threshold = np.percentile(
    train_errors,
    99
)

print(
    "\nInitial anomaly threshold:",
    threshold
)


# ============================================================
# 21. TEST RECONSTRUCTION
# ============================================================

test_predictions = autoencoder.predict(
    X_test_scaled,
    batch_size=512,
    verbose=0
)


# ============================================================
# 22. TEST RECONSTRUCTION ERROR
# ============================================================

test_errors = np.mean(
    np.square(
        X_test_scaled -
        test_predictions
    ),
    axis=1
)


# ============================================================
# 23. INITIAL PREDICTIONS
# ============================================================

initial_predictions = (
    test_errors > threshold
).astype(int)


# ============================================================
# 24. FIND BETTER THRESHOLD USING PR CURVE
# ============================================================

precision, recall, thresholds = precision_recall_curve(
    y_test,
    test_errors
)


# Calculate F1
f1_scores = (
    2 * precision * recall
    /
    (
        precision +
        recall +
        1e-10
    )
)


best_index = np.argmax(
    f1_scores
)


if best_index < len(thresholds):

    best_threshold = thresholds[
        best_index
    ]

else:

    best_threshold = threshold


print(
    "\nBest threshold:",
    best_threshold
)

print(
    "Best F1 score:",
    f1_scores[best_index]
)


# ============================================================
# 25. FINAL PREDICTIONS
# ============================================================

y_pred = (
    test_errors > best_threshold
).astype(int)


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

print(
    "\n=========================================="
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=========================================="
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Fraud"
        ]
    )
)


# ============================================================
# 27. PRECISION / RECALL / F1
# ============================================================

precision_score_value = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall_score_value = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1_score_value = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_errors
)

pr_auc = average_precision_score(
    y_test,
    test_errors
)


print(
    "\nPrecision:",
    precision_score_value
)

print(
    "Recall:",
    recall_score_value
)

print(
    "F1 Score:",
    f1_score_value
)

print(
    "ROC-AUC:",
    roc_auc
)

print(
    "PR-AUC:",
    pr_auc
)


# ============================================================
# 28. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print(
    "\nConfusion Matrix:"
)

print(cm)


plt.figure(figsize=(7, 6))

plt.imshow(
    cm
)

plt.title(
    "Fraud Detection Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    [0, 1],
    ["Normal", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Fraud"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png"
)

plt.close()


# ============================================================
# 29. RECONSTRUCTION ERROR VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

normal_errors = test_errors[
    y_test.values == 0
]

fraud_errors = test_errors[
    y_test.values == 1
]

plt.hist(
    normal_errors,
    bins=50,
    alpha=0.6,
    label="Normal"
)

plt.hist(
    fraud_errors,
    bins=50,
    alpha=0.6,
    label="Fraud"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label="Threshold"
)

plt.title(
    "Reconstruction Error Distribution"
)

plt.xlabel(
    "Reconstruction Error"
)

plt.ylabel(
    "Number of Transactions"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/reconstruction_error.png"
)

plt.close()


# ============================================================
# 30. SAVE THRESHOLD
# ============================================================

joblib.dump(
    best_threshold,
    "models/threshold.pkl"
)


# ============================================================
# 31. SAVE METRICS
# ============================================================

metrics = {
    "precision": precision_score_value,
    "recall": recall_score_value,
    "f1_score": f1_score_value,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "threshold": best_threshold
}

metrics_df = pd.DataFrame(
    [metrics]
)

metrics_df.to_csv(
    "results/metrics.csv",
    index=False
)


# ============================================================
# 32. SAVE TEST PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame({
    "actual": y_test.values,
    "reconstruction_error": test_errors,
    "predicted": y_pred
})

prediction_results.to_csv(
    "results/predictions.csv",
    index=False
)


# ============================================================
# 33. FINAL SUMMARY
# ============================================================

print(
    "\n=========================================="
)

print(
    "DEEPGUARD TRAINING COMPLETE"
)

print(
    "=========================================="
)

print(
    "Model:",
    "models/fraud_autoencoder.keras"
)

print(
    "Scaler:",
    "models/scaler.pkl"
)

print(
    "Threshold:",
    "models/threshold.pkl"
)

print(
    "\nPrecision:",
    round(precision_score_value, 4)
)

print(
    "Recall:",
    round(recall_score_value, 4)
)

print(
    "F1:",
    round(f1_score_value, 4)
)

print(
    "ROC-AUC:",
    round(roc_auc, 4)
)

print(
    "PR-AUC:",
    round(pr_auc, 4)
)

print(
    "\nProject completed successfully!"
)