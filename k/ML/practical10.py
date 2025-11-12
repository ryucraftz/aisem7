# -------------------------------------------------------------
# DATA WRANGLING ON REAL ESTATE MARKET DATA
# -------------------------------------------------------------

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Step 2: Load the Dataset
data = pd.read_csv("RealEstate_Prices.csv")
print("✅ Dataset Loaded Successfully!\n")

print("Shape of Dataset:", data.shape)
print("\nFirst 5 rows:\n", data.head())
print("\nData Info:\n")
print(data.info())

# -------------------------------------------------------------
# Step 3: Clean Column Names
# -------------------------------------------------------------
data.columns = data.columns.str.strip().str.replace('[^A-Za-z0-9]+', '_', regex=True).str.lower()
print("\n✅ Column names cleaned:\n", data.columns)

# -------------------------------------------------------------
# Step 4: Handle Missing Values
# -------------------------------------------------------------
print("\nMissing Values Before:\n", data.isnull().sum())

# Fill numeric columns with mean
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Fill categorical columns with mode
categorical_cols = data.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\n✅ Missing values handled successfully.")

# -------------------------------------------------------------
# Step 5: Example of Data Merging (if another dataset available)
# -------------------------------------------------------------
# For demonstration, let's assume we have a neighborhood demographics dataset
# Uncomment and modify this block if you have another file to merge.

# demographics = pd.read_csv("Neighborhood_Demographics.csv")
# data = pd.merge(data, demographics, on="neighborhood", how="left")
# print("\n✅ Data merged successfully!")

# -------------------------------------------------------------
# Step 6: Filter and Subset Data
# -------------------------------------------------------------
# Example: filter properties sold after 2015 and priced above 50,000
if 'year_sold' in data.columns and 'price' in data.columns:
    data_filtered = data[(data['year_sold'] >= 2015) & (data['price'] > 50000)]
else:
    data_filtered = data.copy()

print("\n✅ Data filtered based on criteria.")
print("Filtered Dataset Shape:", data_filtered.shape)

# -------------------------------------------------------------
# Step 7: Encode Categorical Variables
# -------------------------------------------------------------
encoder = LabelEncoder()
for col in categorical_cols:
    data_filtered[col] = encoder.fit_transform(data_filtered[col])

print("\n✅ Categorical columns encoded using LabelEncoder.")

# -------------------------------------------------------------
# Step 8: Aggregate Data (Summary Statistics)
# -------------------------------------------------------------
if 'neighborhood' in data.columns and 'price' in data.columns:
    neighborhood_summary = data.groupby('neighborhood')['price'].agg(['mean', 'median', 'max', 'min']).reset_index()
    print("\n📊 Average Price by Neighborhood:\n", neighborhood_summary.head())
else:
    print("\n⚠️ 'neighborhood' or 'price' column not found, skipping aggregation.")

# -------------------------------------------------------------
# Step 9: Identify and Handle Outliers (IQR method)
# -------------------------------------------------------------
for col in numeric_cols:
    Q1 = data_filtered[col].quantile(0.25)
    Q3 = data_filtered[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data_filtered[col] = np.where(data_filtered[col] > upper, upper,
                                  np.where(data_filtered[col] < lower, lower, data_filtered[col]))

print("\n✅ Outliers handled using IQR method.")

# -------------------------------------------------------------
# Step 10: Visualization (Optional)
# -------------------------------------------------------------
if 'price' in data_filtered.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=data_filtered['price'])
    plt.title("Boxplot of Property Prices (After Outlier Treatment)")
    plt.show()

# -------------------------------------------------------------
# Step 11: Export Cleaned Dataset
# -------------------------------------------------------------
data_filtered.to_csv("Cleaned_RealEstate_Prices.csv", index=False)
print("\n💾 Cleaned Real Estate Dataset exported successfully!")



##creating a dataset and using it if csv not present

10 practical

# ------------------------------------------------------------
# Project: Data Wrangling on Real Estate Market Dataset
# ------------------------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ------------------------------------------------------------
# 1. Import Dataset and Clean Column Names
# ------------------------------------------------------------

# Load dataset (replace with your file path)
file_path = "RealEstate_Prices.csv"

try:
    df = pd.read_csv(file_path)
except:
    # Simulated dataset for demonstration
    data = {
        'Property ID': [101, 102, 103, 104, 105, 106],
        'Property Type': ['House', 'Apartment', 'House', 'Condo', 'House', 'Apartment'],
        'Location ': ['Downtown', 'Suburb', 'Downtown', 'Suburb', 'Countryside', 'Downtown'],
        'Sale Price ($)': [500000, 300000, 450000, 280000, 150000, 310000],
        'Size (sqft)': [2000, 1200, 1800, 1100, 900, 1250],
        'Bedrooms ': [3, 2, 3, 2, 2, 2],
        'Bathrooms': [2, 1, 2, 1, 1, 2],
        'Year Built': [2010, 2015, 2005, 2012, 1998, 2018],
        'Neighborhood Rating': [8, 7, np.nan, 6, 5, 7]
    }
    df = pd.DataFrame(data)

print("===== Original Dataset Preview =====")
print(df.head(), "\n")

# Clean column names: remove spaces, special chars, lowercase
df.columns = df.columns.str.strip().str.lower().str.replace('[^a-zA-Z0-9]', '_', regex=True)
print("Cleaned Column Names:\n", df.columns, "\n")

# ------------------------------------------------------------
# 2. Handle Missing Values
# ------------------------------------------------------------

print("Missing Values Before:\n", df.isnull().sum(), "\n")

# Fill missing neighborhood rating with median value
df['neighborhood_rating'].fillna(df['neighborhood_rating'].median(), inplace=True)

print("Missing Values After:\n", df.isnull().sum(), "\n")

# ------------------------------------------------------------
# 3. (Optional) Merge with Additional Dataset (if available)
# ------------------------------------------------------------

# Example simulated demographics dataset
demographics = pd.DataFrame({
    'location': ['Downtown', 'Suburb', 'Countryside'],
    'avg_income': [80000, 60000, 40000],
    'school_quality': [9, 7, 5]
})

# Merge datasets based on location
merged_df = pd.merge(df, demographics, left_on='location', right_on='location', how='left')

print("===== After Merging with Demographics =====")
print(merged_df.head(), "\n")

# ------------------------------------------------------------
# 4. Filter & Subset Data
# ------------------------------------------------------------

# Example: Filter properties built after 2010 and located in 'Downtown'
filtered_df = merged_df[(merged_df['year_built'] > 2010) & (merged_df['location'] == 'Downtown')]
print("===== Filtered Data (Downtown & Built after 2010) =====")
print(filtered_df, "\n")

# ------------------------------------------------------------
# 5. Handle Categorical Variables (Encoding)
# ------------------------------------------------------------

# Label Encoding for 'property_type'
le = LabelEncoder()
merged_df['property_type_encoded'] = le.fit_transform(merged_df['property_type'])

# One-hot encoding for 'location'
merged_df = pd.get_dummies(merged_df, columns=['location'], drop_first=True)

print("===== Encoded Dataset Preview =====")
print(merged_df.head(), "\n")

# ------------------------------------------------------------
# 6. Aggregation and Summary Statistics
# ------------------------------------------------------------

# Average sale price by property type
avg_price_by_type = merged_df.groupby('property_type')['sale_price_'].mean().sort_values(ascending=False)
print("Average Sale Price by Property Type:\n", avg_price_by_type, "\n")

# Average price by neighborhood
avg_price_by_location = merged_df.groupby(['property_type_encoded']).agg({'sale_price_':'mean', 'size_sqft':'mean'})
print("Average Price and Size by Property Type:\n", avg_price_by_location, "\n")

# ------------------------------------------------------------
# 7. Identify and Handle Outliers
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))
sns.boxplot(x=merged_df['sale_price_'])
plt.title("Sale Price Distribution (Outlier Detection)")
plt.show()

# Handle outliers using IQR method
Q1 = merged_df['sale_price_'].quantile(0.25)
Q3 = merged_df['sale_price_'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

before = merged_df.shape[0]
merged_df = merged_df[(merged_df['sale_price_'] >= lower_bound) & (merged_df['sale_price_'] <= upper_bound)]
after = merged_df.shape[0]

print(f"Outliers Removed: {before - after}")
print(f"Final Dataset Shape: {merged_df.shape}\n")

# ------------------------------------------------------------
# Visualization – Average Sale Price by Location
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.barplot(x='property_type', y='sale_price_', data=merged_df, estimator=np.mean, palette='viridis')
plt.title('Average Sale Price by Property Type')
plt.ylabel('Average Sale Price ($)')
plt.xlabel('Property Type')
plt.show()

print("✅ Data Wrangling Completed Successfully!")