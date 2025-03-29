import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, SeparableConv1D, MaxPooling1D, Flatten, Dense, Dropout, Add
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def create_convolutional_block(x, filters, kernel_size=3):
    """
    A convolutional block with Conv1D, BatchNormalization, and ReLU activation
    """
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def create_inverted_residual_block(x, filters, kernel_size=3):
    """
    An inverted residual block as described in the paper:
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
    An output block with:
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

def predict_url_class(test_model, url_features, threshold=0.5):
    """
    Predict whether URLs are phishing (1) or benign (0)
    """
    # Get prediction probabilities
    predictions = test_model.predict(url_features)
    print(predictions[:20])
    
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
    df = pd.read_excel("training_feature_dataset.xlsx")


    df = df.drop(columns=["url"])

    X = df.drop(columns=["label"])
    y = df["label"]

    # Convert string labels to numbers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y) 

    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    # Normalize features (MLP performs better with scaled input)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Save the fitted scaler
    joblib.dump(scaler, "scaler.pkl")
    print("Saved Scaler Transform!")

    # Split into training and testing sets (80% train, 20% test)
    X_train, y_train = X_scaled, y
    
    history = model.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.2)

    model.save("phishing_detection_model.h5")
    print("Model has been saved!")

    test_model = load_model("phishing_detection_model.h5")

    # Load dataset
    df = pd.read_excel("testing_feature_dataset.xlsx")


    df = df.drop(columns=["url"])

    X_test = df.drop(columns=["label"])
    y_test = df["label"]

    # Convert string labels to numbers
    label_encoder = LabelEncoder()
    y_test = label_encoder.fit_transform(y_test) 

    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    # Load the saved scaler
    scaler = joblib.load("scaler.pkl")
    X_test_scaled = scaler.transform(X_test)
    
    # Predict on test set
    y_test_pred = predict_url_class(test_model, X_test_scaled, 0.5)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")    

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve, auc, matthews_corrcoef

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # Compute MCC
    mcc = matthews_corrcoef(y_test, y_test_pred)

    # Compute ROC Curve and AUC
    y_prob = model.predict(X_test_scaled)[:, 0]  # Get probability scores for class 0 (phishing)
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=0)
    roc_auc = auc(fpr, tpr)

    # Print MCC
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    # Print Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Phishing"], yticklabels=["Benign", "Phishing"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot ROC Curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

    import matplotlib.pyplot as plt
    # Plot accuracy vs. epochs
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot loss vs. epochs
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid()
    plt.show()
