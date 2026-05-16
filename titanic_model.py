"""
Titanic Survival Prediction — Logistic Regression
===================================================
Full pipeline: preprocessing → feature engineering → training →
evaluation → misclassification analysis → model export.

Usage:
  pip install scikit-learn pandas numpy matplotlib seaborn joblib
  python titanic_model.py
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/yasserh/titanic-dataset
# or: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
df = pd.read_csv('titanic.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

# ─────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n--- Missing Values ---")
print(df.isnull().sum())

print(f"\nOverall survival rate: {df['Survived'].mean():.2%}")
print(f"Survival by sex:\n{df.groupby('Sex')['Survived'].mean()}")
print(f"Survival by class:\n{df.groupby('Pclass')['Survived'].mean()}")

# ─────────────────────────────────────────────────────────────
# 3. PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
age_median = df['Age'].median()

df['Age']      = df['Age'].fillna(age_median)
df['Embarked'] = df['Embarked'].fillna('S')
df['Fare']     = df['Fare'].fillna(df['Fare'].median())

# Encode categorical
df['Sex_enc']    = (df['Sex'] == 'female').astype(int)
df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)

# Derived features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
df['FareLog']    = np.log1p(df['Fare'])

def age_group(a):
    """Bin age into 5 meaningful groups."""
    if a <= 12:  return 0   # Child
    if a <= 18:  return 1   # Teen
    if a <= 35:  return 2   # YoungAdult
    if a <= 60:  return 3   # MiddleAged
    return 4                # Senior

df['AgeGroup'] = df['Age'].apply(age_group)

FEATURES = [
    'Pclass', 'Sex_enc', 'Age', 'SibSp', 'Parch',
    'FareLog', 'Embarked_C', 'Embarked_Q',
    'FamilySize', 'IsAlone', 'AgeGroup'
]
TARGET = 'Survived'

X = df[FEATURES].astype(float)
y = df[TARGET]
assert X.isnull().sum().sum() == 0, "NaN values remain!"
print(f"\nFeature matrix shape: {X.shape}")

# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# 5. TRAIN LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────
model = LogisticRegression(
    C=1.0,             # regularization strength (inverse)
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
)
model.fit(X_train_s, y_train)
print("\nModel trained ✓")

# ─────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm  = confusion_matrix(y_test, y_pred)

print(f"\n{'='*40}")
print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ─────────────────────────────────────────────────────────────
# 7. MISCLASSIFICATION ANALYSIS
# ─────────────────────────────────────────────────────────────
test_df = X_test.copy()
test_df['Survived']  = y_test.values
test_df['Predicted'] = y_pred
test_df['Prob']      = y_prob.round(4)
test_df['Correct']   = y_pred == y_test.values

misclassified = test_df[~test_df['Correct']]
FP = misclassified[misclassified['Survived'] == 0]   # died, predicted survived
FN = misclassified[misclassified['Survived'] == 1]   # survived, predicted died

print(f"\n{'='*40}")
print(f"Total misclassified : {len(misclassified)}")
print(f"False Positives (FP): {len(FP)}  — died but predicted survived")
print(f"False Negatives (FN): {len(FN)}  — survived but predicted died")

print("\n--- FP Profile (died, predicted survived) ---")
print(FP[FEATURES + ['Prob']].describe().round(3))

print("\n--- FN Profile (survived, predicted died) ---")
print(FN[FEATURES + ['Prob']].describe().round(3))

print("\n--- FP vs FN Mean Comparison ---")
compare = pd.DataFrame({
    'FP_mean': FP[FEATURES].mean(),
    'FN_mean': FN[FEATURES].mean(),
    'All_mean': X_test[FEATURES].mean()
}).round(3)
print(compare)

# ─────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE (COEFFICIENTS)
# ─────────────────────────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature':     FEATURES,
    'Coefficient': model.coef_[0],
}).assign(AbsCoef=lambda d: d['Coefficient'].abs()) \
  .sort_values('AbsCoef', ascending=False)

print(f"\n{'='*40}")
print("Feature Coefficients (standardized):")
print(coef_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Titanic — Logistic Regression Analysis', fontsize=14, fontweight='bold')

# Confusion matrix
ConfusionMatrixDisplay(cm, display_labels=['Died','Survived']).plot(ax=axes[0], colorbar=False)
axes[0].set_title('Confusion Matrix')

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], name=f'LR (AUC={auc:.3f})')
axes[1].set_title('ROC Curve')

# Coefficients
colors = ['#2e8b57' if c > 0 else '#b84c2e' for c in coef_df['Coefficient']]
axes[2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
axes[2].axvline(0, color='black', lw=0.8)
axes[2].set_title('Feature Coefficients')
axes[2].set_xlabel('Coefficient (standardized)')

plt.tight_layout()
plt.savefig('titanic_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: titanic_analysis.png")
plt.show()

# ─────────────────────────────────────────────────────────────
# 10. SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────
joblib.dump(model,    'titanic_model.pkl')
joblib.dump(scaler,   'titanic_scaler.pkl')
joblib.dump(FEATURES, 'feature_names.pkl')
print("Model saved: titanic_model.pkl, titanic_scaler.pkl, feature_names.pkl")

# ─────────────────────────────────────────────────────────────
# 11. INFERENCE EXAMPLE (how to use saved model)
# ─────────────────────────────────────────────────────────────
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked,
                     model=model, scaler=scaler):
    """
    Predict survival for a new passenger.
    Returns (prediction, probability).
    """
    sex_enc    = 1 if sex == 'female' else 0
    emb_c      = 1 if embarked == 'C' else 0
    emb_q      = 1 if embarked == 'Q' else 0
    family     = sibsp + parch + 1
    is_alone   = int(family == 1)
    fare_log   = np.log1p(fare)
    age_grp    = age_group(age)

    feats = np.array([[pclass, sex_enc, age, sibsp, parch, fare_log,
                       emb_c, emb_q, family, is_alone, age_grp]])
    feats_s = scaler.transform(feats)
    pred    = model.predict(feats_s)[0]
    prob    = model.predict_proba(feats_s)[0, 1]
    return pred, prob

# Example predictions
examples = [
    (1, 'female', 25, 1, 0, 80, 'C'),   # 1st class young woman → likely survived
    (3, 'male',   35, 0, 0, 8,  'S'),   # 3rd class man alone → likely died
    (2, 'female', 10, 0, 2, 25, 'S'),   # 2nd class girl with family → likely survived
]
print("\n--- Example Predictions ---")
for args in examples:
    pred, prob = predict_survival(*args)
    verdict = 'SURVIVED' if pred else 'DIED'
    print(f"  Class {args[0]}, {args[1]}, age {args[2]} → {verdict} ({prob:.1%} survival prob)")
