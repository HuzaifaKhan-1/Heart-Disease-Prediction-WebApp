import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #for scaling our data
from sklearn.utils import class_weight #for handling unbalanced data
import tensorflow as tf #for building our neural network 
from tensorflow.keras.models import Sequential #for creating the network
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization #Network building blocks
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc #for testing model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pickle

df= pd.read_csv('heart.csv')
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Replace missing values (marked as '?' with NaN and remove them)
df = df.replace('?', np.nan)
df = df.dropna()

# Convert all columns to numbers
for column in df.columns:
    df[column] = pd.to_numeric(df[column])

# Convert target to Critical/Non-Critical
df['target'] = df['target'].map(lambda x: 'Critical' if x>0 else 'Non-Critical')

print(f"Total number of patients: {len(df)}")
print("\nNumber of patients in each category:")
target_dist = df['target'].value_counts()
print(target_dist)

df.info()

df.describe()

# Separate feature (X) and target (y)
X = df.drop('target', axis=1)
y = (df['target'] == 'Critical').astype(int)

# Split into 70:30 for training, and testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size = 0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Scale the data for ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

def create_beginner_model():
    
    # WHAT: Creates a simple neural network for heart disease prediction...
    
    print("Creating our neual network....")
    model = Sequential([
        #First layer - takes in patient data
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
    

    # Middle layer - finds complex patterns
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    # Final layer - makes the prediction (Critical or Non-Critical)
    Dense(1, activation='sigmoid')

    ])

    # Tell the model how to learn
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create our model
model = create_beginner_model()
print("\nOur Neural Network Structure: ")
model.summary()

# Add early stopping to prevent over-training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50, # Number of times to look at all data
    batch_size=32, # Number of patients to look at once
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nPreformance Metrics: ")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Training and validation accuracy
training_accuracy = history.history['accuracy'][-1]
validation_accuracy = history.history['val_accuracy'][-1]

print(f"\nTraining Accuracy: {training_accuracy:.2f}")
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Make predictions
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

# Testing accuracy
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTesting Accuracy: {test_accuracy:.2f}")

# Plotting the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Results of Our Predictions')
plt.xlabel('Actual Diagnosis')
plt.ylabel('Our Model\'s Prediction')
plt.xticks([0.5, 1.5], ['Non-Critical', 'Critical'])
plt.yticks([0.5, 1.5], ['Non-Critical', 'Critical'])
plt.show()


def check_patient_risk(model, scaler, sample_patient):
    """
    Predicts the heart disease risk for a given patient data sample.
    
    Parameters:
        model: Trained neural network model.
        scaler: Fitted StandardScaler object used for scaling data.
        sample_patient: List or array-like of patient data with the same feature order as the training data.
    
    Returns:
        str: Risk level as 'Critical' or 'Non-Critical'.
    """
    # Ensure the sample_patient is reshaped correctly for scaling
    sample_patient_scaled = scaler.transform([sample_patient])  # Transform the data using the trained scaler
    
    # Make a prediction
    prediction_prob = model.predict(sample_patient_scaled)
    prediction = (prediction_prob > 0.5).astype(int)  # Convert probability to binary output
    
    # Interpret and return the result
    risk_level = 'Critical' if prediction[0][0] == 1 else 'Non-Critical'
    print(f"Prediction Probability: {prediction_prob[0][0]:.2f}")
    return risk_level


# Example patient data (replace with actual values)
sample_patient = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]  # Example patient features

# Check the risk for this patient
risk_level = check_patient_risk(model, scaler, sample_patient)
print(f"Risk Level: {risk_level}")


# Plotting Training and Validation Accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# Plotting Training and Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# Plotting ROC Curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# Displaying a Classification Report as Text
report = classification_report(y_test, y_pred, target_names=['Non-Critical', 'Critical'])
print("Classification Report:\n")
print(report)

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, color='purple')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()

# Make pickle file of our model
pickle.dump(model, open("model.pkl", "wb"))