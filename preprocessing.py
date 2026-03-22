# =============================================================================
# EEG-Based Epileptic Seizure Detection
# Step 1: Data Preprocessing Pipeline
# Author: Bhavika
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "data/epilepsy_federated_dataset.csv"   # <-- update path if needed
TARGET_COL = "Seizure_Type_Label"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")

# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

# Basic info
print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Basic Statistics ---")
print(df.describe())

# Missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0]
if missing_df.empty:
    print("No missing values found! ✅")
else:
    print(missing_df)

# Duplicate rows
print("\n--- Duplicate Rows ---")
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Class distribution
print("\n--- Target Class Distribution ---")
class_dist = df[TARGET_COL].value_counts()
print(class_dist)
class_pct = df[TARGET_COL].value_counts(normalize=True) * 100
print("\nClass percentages:")
print(class_pct.round(2))

# =============================================================================
# STEP 3: VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Generating EDA Plots")
print("=" * 60)

# Plot 1: Class distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(x=TARGET_COL, data=df, palette='viridis')
plt.title('Class Distribution (Seizure_Type_Label)', fontsize=14)
plt.xlabel('Seizure Type (0=No Seizure, 1=Focal, 2=Generalized)')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/class_distribution.png")

# Plot 2: Missing values heatmap (if any)
if not missing_df.empty:
    plt.figure(figsize=(12, 4))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/missing_values.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/missing_values.png")

# Plot 3: Correlation heatmap (top 20 features)
plt.figure(figsize=(16, 12))
numeric_df = df.select_dtypes(include=[np.number])
top_features = numeric_df.corr()[TARGET_COL].abs().nlargest(21).index
corr_matrix = numeric_df[top_features].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.3)
plt.title('Correlation Heatmap (Top 20 Features with Target)', fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/correlation_heatmap.png")

# =============================================================================
# STEP 4: HANDLE MISSING VALUES
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Handling Missing Values")
print("=" * 60)

# Fill numerical columns with median
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled '{col}' with median: {median_val:.4f}")

print("Missing value handling complete ✅")

# =============================================================================
# STEP 5: HANDLE DUPLICATES
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Removing Duplicates")
print("=" * 60)

before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)
print(f"Removed {before - after} duplicate rows")
print(f"Dataset size after dedup: {df.shape} ✅")

# =============================================================================
# STEP 6: FEATURE - TARGET SEPARATION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Feature-Target Separation")
print("=" * 60)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Feature columns: {list(X.columns)}")

# =============================================================================
# STEP 7: TRAIN-TEST SPLIT
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Train-Test Split (80-20, Stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

print("\nClass distribution in train set:")
print(y_train.value_counts(normalize=True).round(3) * 100)
print("\nClass distribution in test set:")
print(y_test.value_counts(normalize=True).round(3) * 100)

# =============================================================================
# STEP 8: FEATURE SCALING (StandardScaler)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Feature Scaling (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert back to DataFrame for readability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

print(f"Scaling complete ✅")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape:  {X_test_scaled.shape}")
print(f"\nSample mean after scaling (should be ~0): {X_train_scaled.mean().mean():.6f}")
print(f"Sample std after scaling  (should be ~1): {X_train_scaled.std().mean():.6f}")

# =============================================================================
# STEP 9: SAVE PREPROCESSED DATA
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Saving Preprocessed Data")
print("=" * 60)

X_train_scaled.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
X_test_scaled.to_csv(f"{OUTPUT_DIR}/X_test.csv",   index=False)
y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv",         index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv",           index=False)

print(f"Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv to '{OUTPUT_DIR}/' ✅")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Total samples        : {len(df)}")
print(f"  Total features       : {X.shape[1]}")
print(f"  Target column        : {TARGET_COL}")
print(f"  Classes              : {sorted(y.unique())}")
print(f"  Train size           : {X_train_scaled.shape[0]}")
print(f"  Test size            : {X_test_scaled.shape[0]}")
print(f"  Scaling              : StandardScaler applied")
print(f"  Outputs saved to     : ./{OUTPUT_DIR}/")
print("=" * 60)

