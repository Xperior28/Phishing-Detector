import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_excel("feature_dataset.xlsx")


df = df.drop(columns=["url"])

X = df.drop(columns=["label"])
y = df["label"]

# Convert string labels to numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # "phishing" → 1, "clean" → 0

print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Normalize features (MLP performs better with scaled input)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout

# Define the MLP model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')   # Output layer for binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Predict on test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)   # Convert probabilities to 0 or 1

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
