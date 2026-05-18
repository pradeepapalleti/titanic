"""
Titanic Survival Prediction — Logistic Regression (training only)
===============================================================
Preprocessing → feature engineering → training → evaluation → model export.

Usage:
    pip install scikit-learn pandas numpy
    python titanic_model.py
"""

import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
df = pd.read_csv('titanic.csv')

# Basic preprocessing / feature engineering
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex_enc'] = df['Sex'].map({'male': 0, 'female': 1}).fillna(0).astype(int)
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['FareLog'] = np.log1p(df['Fare'])

emb = pd.get_dummies(df['Embarked'], prefix='Embarked')
for col in ['Embarked_C', 'Embarked_Q', 'Embarked_S']:
    if col not in emb.columns:
        emb[col] = 0
emb = emb[['Embarked_C', 'Embarked_Q', 'Embarked_S']]
df = pd.concat([df, emb], axis=1)

# Family features
df['FamilySize'] = df['SibSp'].fillna(0).astype(int) + df['Parch'].fillna(0).astype(int)
df['IsAlone'] = (df['FamilySize'] == 0).astype(int)

# AgeGroup: simple bins
age_bins = [0, 12, 20, 40, 120]
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=False, include_lowest=True).fillna(2).astype(int)

# Final feature list (matches previous artifact)
FEATURES = [
    'Pclass', 'Sex_enc', 'Age', 'SibSp', 'Parch', 'FareLog',
    'Embarked_C', 'Embarked_Q', 'FamilySize', 'IsAlone', 'AgeGroup'
]

# Ensure no missing values in FEATURES
X = df[FEATURES].copy()
y = df['Survived'].astype(int)

# Drop rows with missing target or feature values
mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask]
y = y[mask]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

# Evaluate
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Trained LogisticRegression — Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")

# Cross-validation (optional, quick)
cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='roc_auc')
print(f"5-fold ROC-AUC: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

# Export model artifact
artifact = {
    'model_type': 'logistic_regression',
    'intercept': float(model.intercept_[0]),
    'coef_values': [float(c) for c in model.coef_[0]],
    'feature_means': [float(m) for m in scaler.mean_],
    'feature_stds': [float(s) for s in np.sqrt(scaler.var_)],
    'features': FEATURES,
    'metrics': {
        'accuracy': float(acc),
        'roc_auc': float(auc)
    }
}

with open('titanic_model.json', 'w') as f:
    json.dump(artifact, f, indent=2)

print('\nModel saved: titanic_model.json')
