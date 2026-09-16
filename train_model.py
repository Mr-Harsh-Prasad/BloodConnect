import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print("Total records:", len(data))

features = [
    "Age",
    "Weight",
    "Hemoglobin",
    "BP_Systolic",
    "BP_Diastolic",
    "Heart_Rate",
    "Temperature_F",
    "Previous_Donation"
]

X = data[features]
y = data["Result"]

numeric_features = [
    "Age",
    "Weight",
    "Hemoglobin",
    "BP_Systolic",
    "BP_Diastolic",
    "Heart_Rate",
    "Temperature_F"
]

categorical_features = [
    "Previous_Donation"
]


# -----------------------------------
# 4. Preprocessing
# -----------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

classifier = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
joblib.dump(model, "model.pkl")

print("\nmodel.pkl created successfully!")
fitted_preprocessor = model.named_steps["preprocessor"]

numeric_scaler = fitted_preprocessor.named_transformers_["numeric"]

joblib.dump(numeric_scaler, "scaler.pkl")

print("scaler.pkl created successfully!")

print("\nTraining process completed successfully.")



''' python train_model.py
key to train the model
'''