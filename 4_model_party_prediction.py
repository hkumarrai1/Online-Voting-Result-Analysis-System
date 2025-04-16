# 1. Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 2. Load cleaned dataset
df = pd.read_csv("cleaned_loksabha_data.csv")

# 3. Show top 5 rows
print(df.head())

# 4. Encode categorical variables
label_encoder = LabelEncoder()
df['party'] = label_encoder.fit_transform(df['party'])  # Encode target variable

# Save the LabelEncoder for the 'party' column
joblib.dump(label_encoder, "party_label_encoder.pkl")  # Save the encoder for later use

# Save the LabelEncoder for the 'state' column
state_encoder = LabelEncoder()
df['state'] = state_encoder.fit_transform(df['state'])  # Encode the 'state' column
joblib.dump(state_encoder, "state_label_encoder.pkl")  # Save the encoder for later use

# Automatically encode all other categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# 5. Split dataset into features (X) and target (y)
X = df.drop(columns=['party'])  # Drop target column
y = df['party']

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train Random Forest Classifier with optimized parameters
model = RandomForestClassifier(
    n_estimators=100,  # Reduce the number of trees (default is 100)
    max_depth=10,      # Limit the depth of each tree
    random_state=42
)
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 10. Save the trained model (optional)
joblib.dump(model, "party_prediction_model.pkl")