# =============================================================================
# EEG-Based Epileptic Seizure Detection
# Step 2: Feature Selection + SMOTE (Class Balancing)
# Author: Bhavika
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH        = "outputs/X_train.csv"
TEST_PATH        = "outputs/X_test.csv"
Y_TRAIN_PATH     = "outputs/y_train.csv"
Y_TEST_PATH      = "outputs/y_test.csv"
OUTPUT_DIR       = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# OUR 23 DOMAIN-SPECIFIC FEATURES
# =============================================================================

SELECTED_FEATURES = [
    # Frequency Domain (6)
    "Gamma_Band_Power",
    "Delta_Band_Power",
    "Low_to_High_Frequency_Power_Ratio",
    "Spectral_Entropy",
    "Theta_Band_Power",
    "Spectral_Edge_Frequency",

    # Entropy & Nonlinear Complexity (5)
    "Sample_Entropy",
    "Higuchi_Fractal_Dimension",
    "Lyapunov_Exponent",
    "Permutation_Entropy",
    "Lempel_Ziv_Complexity",

    # Wavelet (3)
    "Discrete_Wavelet_Transform",
    "Wavelet_Entropy",
    "Wavelet_Energy",

    # Time Domain (4)
    "Cross_Correlation_Between_Channels",
    "Hjorth_Complexity",
    "Zero_Crossing_Rate",
    "Hjorth_Mobility",

    # Clinical Metadata (5)
    "Seizure_Duration",
    "Pre_Seizure_Pattern",
    "Post_Seizure_Recovery",
    "Interictal_Spike_Rate",
    "Seizure_Frequency_Per_Hour",
]

# =============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading Preprocessed Data")
print("=" * 60)

X_train = pd.read_csv(DATA_PATH)
X_test  = pd.read_csv(TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_test  = pd.read_csv(Y_TEST_PATH).squeeze()

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"y_test shape  : {y_test.shape}")

# =============================================================================
# STEP 2: DOMAIN-SPECIFIC FEATURE SELECTION (23 features)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Domain-Specific Feature Selection")
print("=" * 60)

# Check which of our 23 features actually exist in the dataset
available     = [f for f in SELECTED_FEATURES if f in X_train.columns]
not_available = [f for f in SELECTED_FEATURES if f not in X_train.columns]

print(f"\nFeatures found     : {len(available)}")
print(f"Features NOT found : {len(not_available)}")

if not_available:
    print("\nMissing features (check column names in your dataset):")
    for f in not_available:
        print(f"  - {f}")

X_train_selected = X_train[available].copy()
X_test_selected  = X_test[available].copy()

print(f"\nX_train after domain selection : {X_train_selected.shape}")
print(f"X_test after domain selection  : {X_test_selected.shape}")

# =============================================================================
# STEP 3: STATISTICAL VALIDATION WITH SelectKBest
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Statistical Validation with SelectKBest")
print("=" * 60)

# Run SelectKBest on our selected features to get their F-scores
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_train_selected, y_train)

# Build a score dataframe
feature_scores = pd.DataFrame({
    'Feature'  : available,
    'F_Score'  : selector.scores_,
    'P_Value'  : selector.pvalues_
}).sort_values('F_Score', ascending=False).reset_index(drop=True)

print("\nFeature F-Scores (higher = more discriminative):")
print(feature_scores.to_string(index=False))

# Flag any statistically insignificant features (p > 0.05)
weak = feature_scores[feature_scores['P_Value'] > 0.05]
if weak.empty:
    print("\nAll features are statistically significant (p < 0.05) ✅")
else:
    print(f"\nWeak features (p > 0.05) — consider reviewing:")
    print(weak)

# =============================================================================
# STEP 4: VISUALIZE FEATURE IMPORTANCE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Saving Feature Importance Plot")
print("=" * 60)

plt.figure(figsize=(12, 8))
colors = ['#d62728' if s == feature_scores['F_Score'].max()
          else '#1f77b4' for s in feature_scores['F_Score']]
bars = plt.barh(feature_scores['Feature'], feature_scores['F_Score'],
                color=colors)
plt.xlabel('F-Score (ANOVA)', fontsize=12)
plt.title('Feature Importance — Domain Selected Features\n(ANOVA F-Score vs Seizure_Type_Label)',
          fontsize=13)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_selectkbest.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/feature_importance_selectkbest.png")

# =============================================================================
# STEP 5: CLASS IMBALANCE CHECK BEFORE SMOTE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Class Distribution BEFORE SMOTE")
print("=" * 60)

before = y_train.value_counts().sort_index()
print(before)
print(f"\nImbalance ratio: {before.max() / before.min():.2f}x")

# =============================================================================
# STEP 6: APPLY SMOTE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Applying SMOTE")
print("=" * 60)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_selected, y_train)

print(f"X_train after SMOTE : {X_train_smote.shape}")
print(f"y_train after SMOTE : {y_train_smote.shape}")

after = pd.Series(y_train_smote).value_counts().sort_index()
print("\nClass Distribution AFTER SMOTE:")
print(after)
print(f"\nImbalance ratio after SMOTE: {after.max() / after.min():.2f}x ✅")

# =============================================================================
# STEP 7: VISUALIZE CLASS DISTRIBUTION BEFORE VS AFTER
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Saving Class Distribution Plot")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(['No Seizure', 'Focal', 'Generalized'],
            before.values, color=['#1f77b4', '#2ca02c', '#d62728'])
axes[0].set_title('Before SMOTE', fontsize=13)
axes[0].set_ylabel('Count')
for i, v in enumerate(before.values):
    axes[0].text(i, v + 500, str(v), ha='center', fontsize=10)

axes[1].bar(['No Seizure', 'Focal', 'Generalized'],
            after.values, color=['#1f77b4', '#2ca02c', '#d62728'])
axes[1].set_title('After SMOTE', fontsize=13)
axes[1].set_ylabel('Count')
for i, v in enumerate(after.values):
    axes[1].text(i, v + 500, str(v), ha='center', fontsize=10)

plt.suptitle('Class Distribution Before vs After SMOTE', fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/smote_class_distribution.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/smote_class_distribution.png")

# =============================================================================
# STEP 8: SAVE FINAL DATA
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Saving Final Data")
print("=" * 60)

# Convert back to DataFrame
X_train_final = pd.DataFrame(X_train_smote, columns=available)
X_test_final  = X_test_selected.copy()
y_train_final = pd.Series(y_train_smote, name='Seizure_Type_Label')
y_test_final  = y_test.copy()

X_train_final.to_csv(f"{OUTPUT_DIR}/X_train_final.csv", index=False)
X_test_final.to_csv(f"{OUTPUT_DIR}/X_test_final.csv",   index=False)
y_train_final.to_csv(f"{OUTPUT_DIR}/y_train_final.csv", index=False)
y_test_final.to_csv(f"{OUTPUT_DIR}/y_test_final.csv",   index=False)

print(f"Saved X_train_final.csv  : {X_train_final.shape}")
print(f"Saved X_test_final.csv   : {X_test_final.shape}")
print(f"Saved y_train_final.csv  : {y_train_final.shape}")
print(f"Saved y_test_final.csv   : {y_test_final.shape}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PREPROCESSING PART 2 COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Original features         : 52")
print(f"  Domain selected features  : {len(available)}")
print(f"  Feature selection method  : Domain knowledge + SelectKBest validation")
print(f"  Class balancing           : SMOTE applied")
print(f"  Train size before SMOTE   : {len(y_train)}")
print(f"  Train size after SMOTE    : {len(y_train_smote)}")
print(f"  Test size (unchanged)     : {len(y_test_final)}")
print(f"  Outputs saved to          : ./{OUTPUT_DIR}/")
print("=" * 60)
