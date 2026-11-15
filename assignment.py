"""
Machine Learning Assignments
  1. Linear Regression on housing data (predict price)
  2. Logistic Regression on Titanic data (predict survival)
  3. Mini Project 2: House Price Prediction (R² evaluation + predicted vs actual plot)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
)

HOUSING_FILE = "housing.csv"
TITANIC_FILE = "titanic_train.csv"   # change to your Titanic file name if different
HOUSING_TARGET = "price"             # change if your price column has another name


# ============================================================
# PART 1 & 3: HOUSE PRICE PREDICTION (LINEAR REGRESSION)
# ============================================================
def house_price_prediction(path, target):
    print("=" * 60)
    print("PART 1 & 3: HOUSE PRICE PREDICTION (Linear Regression)")
    print("=" * 60)

    # Load data
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {df.columns.tolist()}")

    # Remove ID-like columns and separate features and target
    df = df.drop(columns=[c for c in df.columns if c.lower() == "id"], errors="ignore")
    y = df[target]
    X = df.drop(columns=[target])

    # Convert text columns to numbers (one-hot encoding)
    X = pd.get_dummies(X, drop_first=True, dtype=int)

    # Split first, then fill missing values using only training-set medians (avoids data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nModel trained on", X_train.shape[0], "samples with", X_train.shape[1], "features.")

    # Predict house prices
    y_pred = model.predict(X_test)
    print("\nSample predictions (actual vs predicted):")
    comparison = pd.DataFrame({"Actual": y_test.values[:5], "Predicted": y_pred[:5].round(2)})
    print(comparison.to_string(index=False))

    # Evaluate
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nR² score: {r2:.4f}")
    print(f"RMSE:     {rmse:,.2f}")
    print(f"MAE:      {mae:,.2f}")

    # Plot predicted vs actual
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Predicted vs Actual House Prices (R² = {r2:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("house_predicted_vs_actual.png", dpi=150)
    plt.show()
    print("Plot saved as house_predicted_vs_actual.png\n")

    return model, r2


# ============================================================
# PART 2: TITANIC SURVIVAL PREDICTION (LOGISTIC REGRESSION)
# ============================================================
def titanic_survival_prediction(path):
    print("=" * 60)
    print("PART 2: TITANIC SURVIVAL PREDICTION (Logistic Regression)")
    print("=" * 60)

    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Encode categorical columns as numbers
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int)

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    features += [c for c in df.columns if c.startswith("Embarked_")]

    X = df[features]
    y = df["Survived"]

    # Split (stratify keeps the survival ratio the same in train and test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Did not survive", "Survived"]))

    print("Feature coefficients:")
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": clf.coef_[0].round(4)})
    print(coef_df.to_string(index=False))
    print()

    return clf


# ============================================================
# RUN EVERYTHING
# ============================================================
if __name__ == "__main__":
    house_model, house_r2 = house_price_prediction(HOUSING_FILE, HOUSING_TARGET)
    titanic_model = titanic_survival_prediction(TITANIC_FILE)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"House price model R²:      {house_r2:.4f}")
    print("Titanic model:             Logistic Regression trained and evaluated (see accuracy above)")