import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, SeparableConv1D, MaxPooling1D, Flatten, Dense, Dropout, Add
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_convolutional_block(x, filters, kernel_size=3):
    """
    Create a convolutional block with Conv1D, BatchNormalization, and ReLU activation
    """
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def create_inverted_residual_block(x, filters, kernel_size=3):
    """
    Create an inverted residual block as described in the paper:
    - Convolution
    - Separable Convolution
    - Batch Normalization
    - Activation
    - Convolution
    - Batch Normalization
    - Activation
    """
    # Store input for the skip connection
    skip = x
    
    # First convolution
    x = Conv1D(filters, 1, padding='same')(x)
    
    # Separable convolution (depth-wise + point-wise)
    x = SeparableConv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second convolution
    x = Conv1D(filters, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Add skip connection (residual connection)
    x = Add()([x, skip])
    
    return x

def create_output_block(x, num_classes=2):
    """
    Create output block with:
    - Max Pooling
    - Flatten
    - Dense Layer
    - Dropout
    - Final Dense Layer with Softmax
    """
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    return x

def build_phishing_detection_model(input_shape=(41, 1), num_filters=64, num_classes=2):
    """
    Build the complete phishing detection model with residual pipeline
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial convolutional block
    x = create_convolutional_block(inputs, num_filters)
    
    # Seven inverted residual blocks as mentioned in the paper
    for _ in range(7):
        x = create_inverted_residual_block(x, num_filters)
    
    # Final convolutional block after residual blocks
    x = create_convolutional_block(x, num_filters)
    
    # Output block
    outputs = create_output_block(x, num_classes)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_url_features(url_features):
    """
    Preprocess URL features for the model
    - Reshape to add channel dimension
    """
    return np.reshape(url_features, (url_features.shape[0], url_features.shape[1], 1))

def predict_url_class(model, url_features, threshold=0.5):
    """
    Predict whether URLs are phishing (1) or benign (0)
    """
    # Get prediction probabilities
    predictions = model.predict(url_features)
    
    # Apply threshold (as mentioned in the paper)
    predicted_classes = (predictions[:, 1] >= threshold).astype(int)
    
    return predicted_classes

# Example usage
if __name__ == "__main__":
    # Build the model
    model = build_phishing_detection_model(input_shape=(41, 1))
    
    # Print model summary
    model.summary()

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
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.35, random_state=42)
    
    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.35)
    
    # Predict on test set
    y_pred = predict_url_class(model, X_test, 0.5)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")    