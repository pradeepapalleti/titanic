"""
Titanic Survival Prediction — Logistic Regression
===================================================
Full pipeline: preprocessing → feature engineering → training →
evaluation → model export.

Usage:
    pip install scikit-learn pandas numpy
    python titanic_model.py
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/yasserh/titanic-dataset
# or: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
df = pd.read_csv('titanic.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

# ─────────────────────────────────────────────────────────────
# 2. QUICK DATA PREP
# ─────────────────────────────────────────────────────────────
print("\nPreprocessing data...")

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Encode & create features
df['Sex_enc'] = (df['Sex'] == 'female').astype(int)
df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['FareLog'] = np.log1p(df['Fare'])
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)

FEATURES = ['Pclass', 'Sex_enc', 'Age', 'SibSp', 'Parch', 'FareLog', 
            'Embarked_C', 'Embarked_Q', 'FamilySize', 'IsAlone', 'AgeGroup']

X = df[FEATURES].astype(float)
y = df['Survived']
print(f"Data shape: {X.shape}")

# ─────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# 4. TRAIN & EVALUATE MODEL
# ─────────────────────────────────────────────────────────────
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"\n--- Model Performance ---")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC:  {auc:.4f}")
print(f"\nConfusion Matrix:\n{cm}\n{classification_report(y_test, y_pred)}")

# ─────────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE & SAVE MODEL
# ─────────────────────────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature': FEATURES,
    'Coefficient': model.coef_[0],
    'AbsCoef': np.abs(model.coef_[0]),
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\nTop Features:\n{coef_df.to_string(index=False)}")

# Save model
model_data = {
    'model_type': 'logistic_regression',
    'intercept': float(model.intercept_[0]),
    'coef_values': model.coef_[0].tolist(),
    'feature_means': scaler.mean_.tolist(),
    'feature_stds': scaler.scale_.tolist(),
    'features': FEATURES,
    'metrics': {'accuracy': float(acc), 'roc_auc': float(auc)},
}
with open('titanic_model.json', 'w') as f:
    json.dump(model_data, f, indent=2)
print("\nModel saved: titanic_model.json")

# ─────────────────────────────────────────────────────────────
# 8B. GENERATE COMPREHENSIVE VISUALIZATIONS
# ─────────────────────────────────────────────────────────────

# 1. Main Analysis (Confusion Matrix, ROC Curve, Feature Coefficients)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Titanic — Logistic Regression Analysis', fontsize=16, fontweight='bold')

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=axes[0],
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'],
            cbar=False, annot_kws={'size': 12})
axes[0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True label')
axes[0].set_xlabel('Predicted label')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='#0088cc', lw=2, label=f'LR (AUC={auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)
axes[1].set_xlim([-0.02, 1.02])
axes[1].set_ylim([-0.02, 1.02])
axes[1].set_xlabel('False Positive Rate (Positive label: 1)')
axes[1].set_ylabel('True Positive Rate (Positive label: 1)')
axes[1].set_title('ROC Curve', fontsize=12, fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

# Feature Coefficients
coef_sorted = coef_df.sort_values('Coefficient')
colors = ['#d7553a' if x < 0 else '#2ca02c' for x in coef_sorted['Coefficient']]
axes[2].barh(coef_sorted['Feature'], coef_sorted['Coefficient'], color=colors)
axes[2].set_xlabel('Coefficient (standardized)')
axes[2].set_title('Feature Coefficients', fontsize=12, fontweight='bold')
axes[2].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[2].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('titanic_analysis.png', dpi=150, bbox_inches='tight')
print("[OK] Visualization saved: titanic_analysis.png")
plt.close()

# 2. Feature Distributions
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Feature Distributions by Survival Status', fontsize=16, fontweight='bold')
axes = axes.flatten()

for idx, feature in enumerate(FEATURES):
    axes[idx].hist(X_test[X_test.index.isin(y_test[y_test==0].index)][feature], 
                   alpha=0.6, label='Died', color='#d7553a', bins=20)
    axes[idx].hist(X_test[X_test.index.isin(y_test[y_test==1].index)][feature], 
                   alpha=0.6, label='Survived', color='#2ca02c', bins=20)
    axes[idx].set_title(feature, fontweight='bold')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
print("[OK] Visualization saved: feature_distributions.png")
plt.close()

# 3. Boxplots for Key Features
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Feature Boxplots by Survival Status', fontsize=16, fontweight='bold')
axes = axes.flatten()

key_features = ['Age', 'FareLog', 'Pclass', 'FamilySize', 'AgeGroup']
for idx, feature in enumerate(key_features):
    survived_data = X_test[X_test.index.isin(y_test[y_test==1].index)][feature]
    died_data = X_test[X_test.index.isin(y_test[y_test==0].index)][feature]
    
    bp = axes[idx].boxplot([died_data, survived_data], labels=['Died', 'Survived'],
                           patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#d7553a', '#2ca02c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[idx].set_title(feature, fontweight='bold')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('boxplots.png', dpi=150, bbox_inches='tight')
print("[OK] Visualization saved: boxplots.png")
plt.close()

# 4. Feature Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 10))
correlation_matrix = df[FEATURES + ['Survived']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=ax, square=True, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('feature_heatmap.png', dpi=150, bbox_inches='tight')
print("[OK] Visualization saved: feature_heatmap.png")
plt.close()

# 5. Confusion Matrices at Different Thresholds
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices at Different Prediction Thresholds', fontsize=14, fontweight='bold')

thresholds = [0.3, 0.5, 0.7]
for idx, threshold in enumerate(thresholds):
    y_pred_thresh = (y_prob >= threshold).astype(int)
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    
    sns.heatmap(cm_thresh, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'],
                cbar=False, annot_kws={'size': 12})
    axes[idx].set_title(f'Threshold = {threshold}', fontweight='bold')
    axes[idx].set_ylabel('True label')
    axes[idx].set_xlabel('Predicted label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
print("[OK] Visualization saved: confusion_matrices.png")
plt.close()

# 6. Prediction Probability Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_prob[y_test==0], alpha=0.6, label='Died (actual)', color='#d7553a', bins=30)
ax.hist(y_prob[y_test==1], alpha=0.6, label='Survived (actual)', color='#2ca02c', bins=30)
ax.set_xlabel('Predicted Probability of Survival', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Output 1.png', dpi=150, bbox_inches='tight')
print("[OK] Visualization saved: Output 1.png")
plt.close()

# 7. Model Performance Metrics Summary
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

metrics_text = f"""
MODEL PERFORMANCE SUMMARY
{'='*50}

Accuracy:           {acc:.4f}  ({acc*100:.2f}%)
ROC-AUC:           {auc:.4f}

Confusion Matrix:
  True Negatives:   {cm[0,0]}
  False Positives:  {cm[0,1]}
  False Negatives:  {cm[1,0]}
  True Positives:   {cm[1,1]}

Classification Rates:
  Sensitivity:      {cm[1,1]/(cm[1,0]+cm[1,1]):.4f}
  Specificity:      {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}
  Precision:        {cm[1,1]/(cm[1,1]+cm[0,1]):.4f}

Dataset Split:
  Training samples: {len(X_train)}
  Testing samples:  {len(X_test)}
"""

ax.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig('Output 2.png', dpi=150, bbox_inches='tight')
print("[OK] Visualization saved: Output 2.png")
plt.close()

# 8. Feature Importance Ranked
fig, ax = plt.subplots(figsize=(10, 8))
coef_sorted = coef_df.sort_values('AbsCoef', ascending=True)
colors = ['#d7553a' if x < 0 else '#2ca02c' for x in coef_sorted['Coefficient']]
ax.barh(coef_sorted['Feature'], coef_sorted['Coefficient'], color=colors)
ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance (Ranked by Absolute Coefficient)', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('Output 3.png', dpi=150, bbox_inches='tight')
print("[OK] Visualization saved: Output 3.png")
plt.close()

# 9. Cross-Validation Scores
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='roc_auc')

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(1, len(cv_scores)+1), cv_scores, color='#0088cc', alpha=0.7, edgecolor='black')
ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
ax.set_title('5-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')
ax.set_ylim([0.7, 1.0])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('Output 4.png', dpi=150, bbox_inches='tight')
print("[OK] Visualization saved: Output 4.png")
plt.close()

# 9. Confusion Matrix Breakdown
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['True\nNegatives', 'False\nPositives', 'False\nNegatives', 'True\nPositives']
counts = [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
colors_cm = ['#2ca02c', '#d7553a', '#d7553a', '#2ca02c']
ax.bar(categories, counts, color=colors_cm, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Count', fontsize=11, fontweight='bold')
ax.set_title('Confusion Matrix Breakdown', fontsize=12, fontweight='bold')
for i, v in enumerate(counts):
    ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('Output 5.png', dpi=150, bbox_inches='tight')
print("[OK] Visualization saved: Output 5.png")
plt.close()

# 10. Dataset Pairplot (Reduced features for clarity)
pairplot_features = ['Age', 'Pclass', 'Sex_enc', 'FareLog', 'Survived']
pairplot_df = df[pairplot_features].copy()
pairplot_df['Survived'] = pairplot_df['Survived'].map({0: 'Died', 1: 'Survived'})

pp = sns.pairplot(pairplot_df, hue='Survived', palette={'Died': '#d7553a', 'Survived': '#2ca02c'},
                  diag_kind='hist', plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.6})
pp.fig.suptitle('Dataset Pairplot (Selected Features)', fontsize=14, fontweight='bold', y=1.001)
plt.tight_layout()
plt.savefig('dataset_pairplot.png', dpi=150, bbox_inches='tight')
print("[OK] Visualization saved: dataset_pairplot.png")
plt.close()

# 11. Scaling Comparison (Before vs After)
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Feature Scaling Comparison (Before vs After StandardScaler)', fontsize=14, fontweight='bold')

sample_features = ['Age', 'FareLog', 'Pclass', 'FamilySize', 'AgeGroup', 'Sex_enc']
for idx, feature in enumerate(sample_features):
    row = idx // 3
    col = idx % 3
    
    # Original values
    axes[0, col].hist(X_train[feature], bins=30, alpha=0.7, color='#0088cc', edgecolor='black')
    axes[0, col].set_title(f'{feature} (Original)', fontweight='bold')
    axes[0, col].set_ylabel('Frequency')
    axes[0, col].grid(True, alpha=0.3)
    
    # Scaled values
    axes[1, col].hist(X_train_s[:, FEATURES.index(feature)], bins=30, alpha=0.7, color='#2ca02c', edgecolor='black')
    axes[1, col].set_title(f'{feature} (Scaled)', fontweight='bold')
    axes[1, col].set_ylabel('Frequency')
    axes[1, col].grid(True, alpha=0.3)

axes[0, 0].set_ylabel('Frequency (Original)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Frequency (Scaled)', fontsize=11, fontweight='bold')

print(f"\n{'='*50}")
print("All visualizations generated successfully!")
print(f"{'='*50}")

