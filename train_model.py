import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn.metrics import accuracy_score # Import accuracy_score

# Load dataset
df = pd.read_csv("Training.csv")

# Check if 'prognosis' column exists and handle potential trailing spaces
if 'prognosis' not in df.columns:
    # Check for columns with potential trailing spaces
    potential_cols = [col for col in df.columns if 'prognosis' in col.strip()]
    if potential_cols:
        prognosis_col = potential_cols[0]
        print(f"Warning: Using column '{prognosis_col}' as target. Consider renaming it to 'prognosis'.")
    else:
        raise ValueError("Target column 'prognosis' not found in the dataset.")
else:
    prognosis_col = 'prognosis'

# Drop target column and any potential empty/unnamed columns at the end
X = df.drop(columns=[prognosis_col])
X = X.loc[:, ~X.columns.str.contains('^Unnamed')] # Remove unnamed columns if they exist
y = df[prognosis_col]


# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Split data into Training and Testing sets ---
# Use 80% for training and 20% for testing
# stratify=y_encoded ensures the proportion of each class is maintained in both sets
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# --- Train the model on the Training data ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate the model on the Testing data ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- Print the accuracy ---
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}") # Format to 4 decimal places

# --- Optional: Re-train on the FULL dataset before saving (common practice) ---
# This uses all available data for the final saved model
print("\nRe-training model on the full dataset before saving...")
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X, y_encoded) # Train on original full X and y_encoded

# --- Save the final model (trained on full data), symptoms list, and label encoder ---
with open("model.pkl", "wb") as f:
    # Save the final_model trained on all data
    pickle.dump((final_model, list(X.columns), le), f)

print("Final Random Forest model trained on full data and saved as model.pkl")