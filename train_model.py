import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier


# create folder for trained artifacts
os.makedirs("artifacts", exist_ok=True)


# load dataset
df = pd.read_csv("data/dog_disease.csv")

# remove duplicates
df = df.drop_duplicates()

target = "disease"

X = df.drop(columns=[target])
y = df[target]


# symptom columns (yes/no)
binary_cols = [
    "appetite_loss",
    "vomiting",
    "diarrhea",
    "lethargy",
    "coughing",
    "nasal_discharge",
    "weight_loss",
    "excessive_salivation",
    "seizures"
]

# numeric column
numeric_cols = ["age"]

# categorical columns
nominal_cols = [
    "breed_size",
    "vaccination_status"
]


# convert yes/no to 1/0
for col in binary_cols:
    X[col] = X[col].map({"yes": 1, "no": 0})


# preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols + binary_cols),
        ("nom", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
    ]
)


# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# Decision Tree
dt = Pipeline([
    ("prep", preprocessor),
    ("model", DecisionTreeClassifier())
])

# Random Forest
rf = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier())
])

# KNN
knn = Pipeline([
    ("prep", preprocessor),
    ("scaler", StandardScaler(with_mean=False)),
    ("model", KNeighborsClassifier())
])


# hyperparameter tuning
rf_params = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20]
}

knn_params = {
    "model__n_neighbors": [3, 5, 7, 9],
    "model__weights": ["uniform", "distance"]
}

rf_search = RandomizedSearchCV(rf, rf_params, n_iter=5, cv=3, n_jobs=-1)
knn_search = RandomizedSearchCV(knn, knn_params, n_iter=5, cv=3, n_jobs=-1)

rf_search.fit(X_train, y_train)
knn_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_
best_knn = knn_search.best_estimator_

dt.fit(X_train, y_train)


# hybrid ensemble
hybrid = VotingClassifier(
    estimators=[
        ("dt", dt),
        ("rf", best_rf),
        ("knn", best_knn)
    ],
    voting="soft"
)

hybrid.fit(X_train, y_train)


# evaluate models
models = {
    "Decision Tree": dt,
    "Random Forest": best_rf,
    "KNN": best_knn,
    "Hybrid": hybrid
}

results = []

for name, model in models.items():

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    report = classification_report(
        y_test,
        preds,
        output_dict=True,
        zero_division=0
    )

    results.append([
        name,
        acc,
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
        report["macro avg"]["f1-score"]
    ])

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
)

print("\nMODEL COMPARISON\n")
print(results_df.sort_values(by="Accuracy", ascending=False))


# final model
best_model = hybrid

print("\nFINAL MODEL USED: Hybrid Ensemble")


# confusion matrix
cm = confusion_matrix(
    y_test,
    best_model.predict(X_test)
)

print("\nConfusion Matrix:\n", cm)


# save artifacts
joblib.dump(best_model, "artifacts/disease_prediction_model.pkl")
joblib.dump(preprocessor, "artifacts/preprocessing_pipeline.pkl")

print("\nArtifacts saved successfully.")