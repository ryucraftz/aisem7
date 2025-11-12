# -------------------------------------------------------------
# TELECOM CUSTOMER CHURN DATA CLEANING & PREPARATION
# -------------------------------------------------------------
#only on csv


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 2: Load Dataset
data = pd.read_csv("Telecom_Customer_Churn.csv")
print("✅ Dataset Loaded Successfully!\n")

# Step 3: Explore Dataset
print("Shape of Dataset:", data.shape)
print("\nFirst 5 rows:\n", data.head())
print("\nData Info:\n")
print(data.info())
print("\nSummary Statistics:\n", data.describe())

# Step 4: Check for Missing Values
print("\nMissing Values per Column:\n", data.isnull().sum())

# Fill missing numeric columns with mean
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Fill missing categorical columns with mode
categorical_cols = data.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\n✅ Missing values handled.")

# Step 5: Remove Duplicate Records
before = data.shape[0]
data.drop_duplicates(inplace=True)
after = data.shape[0]
print(f"\nRemoved {before - after} duplicate rows.")

# Step 6: Standardize Inconsistent Formatting (Example: Gender)
if 'gender' in data.columns:
    data['gender'] = data['gender'].str.strip().str.lower()
    data['gender'] = data['gender'].replace({'male': 'Male', 'female': 'Female'})
print("\n✅ Data formatting standardized.")

# Step 7: Convert Columns to Correct Data Types
if 'TotalCharges' in data.columns:
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())
print("\n✅ Columns converted to correct data types.")

# Step 8: Identify and Handle Outliers using IQR
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = np.where(data[col] > upper, upper,
                         np.where(data[col] < lower, lower, data[col]))

print("\n✅ Outliers handled using IQR method.")

# Step 9: Feature Engineering
if 'tenure' in data.columns and 'MonthlyCharges' in data.columns:
    data['TotalSpent'] = data['tenure'] * data['MonthlyCharges']
print("\n✅ Feature Engineering Done (Added TotalSpent).")

# Step 10: Normalize Numeric Data
scaler = MinMaxScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
print("\n✅ Numeric data normalized using MinMaxScaler.")

# Step 11: Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap - Telecom Churn Data")
plt.show()

# Step 12: Split the Dataset into Training and Testing Sets
# (Assuming 'Churn' column is the target variable)
if 'Churn' in data.columns:
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n✅ Data Split Successful!")
    print("Training Set Shape:", X_train.shape)
    print("Testing Set Shape:", X_test.shape)
else:
    print("\n⚠️ No 'Churn' column found. Skipping train-test split.")

# Step 13: Export Cleaned Dataset
data.to_csv("Cleaned_Telecom_Customer_Churn.csv", index=False)
X_train.to_csv("Train_Features.csv", index=False)
X_test.to_csv("Test_Features.csv", index=False)
y_train.to_csv("Train_Labels.csv", index=False)
y_test.to_csv("Test_Labels.csv", index=False)

print("\n💾 Cleaned and split datasets exported successfully!")


#creating a dataset and using it if csv not present

# ------------------------------------------------------------
# Project: Data Cleaning & Preparation - Customer Churn Analysis
# ------------------------------------------------------------

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 2. Import the Dataset
# ------------------------------------------------------------
file_path = "Telecom_Customer_Churn.csv"  # Replace with actual file path
try:
    df = pd.read_csv(file_path)
except:
    # Simulated dataset if actual file not present
    data = {
        'customerID': ['0001', '0002', '0003', '0004', '0005'],
        'gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
        'SeniorCitizen': [0, 1, 0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'No'],
        'Dependents': ['No', 'No', 'Yes', 'No', 'No'],
        'tenure': [1, 34, 2, 45, 5],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No', 'Yes', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'Fiber optic', 'DSL'],
        'OnlineSecurity': ['No', 'Yes', 'No', 'No', 'Yes'],
        'OnlineBackup': ['Yes', 'No', 'No', 'Yes', 'No'],
        'DeviceProtection': ['No', 'Yes', 'No', 'Yes', 'No'],
        'TechSupport': ['No', 'No', 'No', 'Yes', 'Yes'],
        'StreamingTV': ['No', 'Yes', 'No', 'Yes', 'No'],
        'StreamingMovies': ['No', 'Yes', 'No', 'Yes', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check', 'Bank transfer', 'Credit card'],
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70],
        'TotalCharges': ['29.85', '1889.50', '108.15', '1840.75', '151.65'],
        'Churn': ['No', 'No', 'Yes', 'No', 'Yes']
    }
    df = pd.DataFrame(data)

print("===== Dataset Preview =====")
print(df.head(), "\n")

# ------------------------------------------------------------
# 3. Explore the Dataset
# ------------------------------------------------------------
print("Dataset Info:")
print(df.info(), "\n")

print("Missing Values:\n", df.isnull().sum(), "\n")

print("Basic Statistics:")
print(df.describe(include='all').T, "\n")

# ------------------------------------------------------------
# 4. Handle Missing Values
# ------------------------------------------------------------

# Check for blanks or missing TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# ------------------------------------------------------------
# 5. Remove Duplicate Records
# ------------------------------------------------------------
duplicates = df.duplicated().sum()
print(f"Duplicate Rows Found: {duplicates}")
df.drop_duplicates(inplace=True)

# ------------------------------------------------------------
# 6. Fix Inconsistent Formatting
# ------------------------------------------------------------

# Standardize text case for categorical columns
for col in df.select_dtypes(include='object'):
    df[col] = df[col].str.strip().str.lower()

# Example: Fix inconsistent spelling
df['internetservice'].replace({'fiber optics': 'fiber optic'}, inplace=True)

# ------------------------------------------------------------
# 7. Convert Columns to Correct Data Types
# ------------------------------------------------------------
df['SeniorCitizen'] = df['SeniorCitizen'].astype('int')
df['tenure'] = df['tenure'].astype('int')

# ------------------------------------------------------------
# 8. Identify and Handle Outliers
# ------------------------------------------------------------

plt.figure(figsize=(8,4))
sns.boxplot(x=df['MonthlyCharges'])
plt.title("Monthly Charges - Outlier Check")
plt.show()

# Remove extreme outliers beyond 99th percentile
upper_limit = df['MonthlyCharges'].quantile(0.99)
df = df[df['MonthlyCharges'] <= upper_limit]

# ------------------------------------------------------------
# 9. Feature Engineering
# ------------------------------------------------------------

# Create new features
df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)  # Avoid divide by zero
df['senior_flag'] = np.where(df['SeniorCitizen'] == 1, 1, 0)

# Tenure group
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                            labels=['0-12 months', '12-24 months', '24-48 months', '48-72 months'])

# ------------------------------------------------------------
# 10. Encode and Scale Data
# ------------------------------------------------------------

# Label encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# Normalize numerical features
scaler = StandardScaler()
num_cols = ['MonthlyCharges', 'TotalCharges', 'avg_monthly_charge']
df[num_cols] = scaler.fit_transform(df[num_cols])

# ------------------------------------------------------------
# 11. Split Dataset into Training and Testing Sets
# ------------------------------------------------------------
X = df.drop(['churn', 'customerid'], axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# ------------------------------------------------------------
# 12. Export Cleaned Dataset
# ------------------------------------------------------------
df.to_csv("Cleaned_Telecom_Customer_Churn.csv", index=False)
print("\n✅ Cleaned dataset exported as 'Cleaned_Telecom_Customer_Churn.csv'")