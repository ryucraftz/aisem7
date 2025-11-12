# ------------------------------------------------------------
# Project: Data Aggregation - Sales Performance by Region
# ------------------------------------------------------------

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Import the Dataset
# ------------------------------------------------------------

file_path = "Retail_Sales_Data.csv"

try:
    df = pd.read_csv(file_path)
except:
    # Create a sample dataset if not found
    data = {
        'Transaction_Date': pd.date_range(start='2024-01-01', periods=12, freq='M'),
        'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South', 'East', 'West'],
        'Product_Category': ['Electronics', 'Clothing', 'Groceries', 'Furniture',
                             'Clothing', 'Electronics', 'Groceries', 'Furniture',
                             'Furniture', 'Groceries', 'Clothing', 'Electronics'],
        'Quantity_Sold': [100, 200, 150, 80, 180, 220, 160, 90, 120, 210, 190, 250],
        'Sales_Amount': [20000, 15000, 12000, 10000, 22000, 18000, 14000, 11000, 23000, 19000, 16000, 25000]
    }
    df = pd.DataFrame(data)

print("===== Dataset Preview =====")
print(df.head(), "\n")

# ------------------------------------------------------------
# 2. Explore the Dataset
# ------------------------------------------------------------
print("Dataset Info:")
print(df.info(), "\n")

print("Summary Statistics:")
print(df.describe(), "\n")

print("Missing Values:\n", df.isnull().sum(), "\n")

# ------------------------------------------------------------
# 3. Identify Relevant Variables
# ------------------------------------------------------------
# We’ll use Region, Product_Category, and Sales_Amount for aggregation
columns = ['Region', 'Product_Category', 'Sales_Amount']
print("Relevant Columns for Aggregation:", columns, "\n")

# ------------------------------------------------------------
# 4. Aggregate Data by Region
# ------------------------------------------------------------
region_sales = df.groupby('Region')['Sales_Amount'].sum().sort_values(ascending=False)
print("===== Total Sales by Region =====")
print(region_sales, "\n")

# ------------------------------------------------------------
# 5. Visualize Sales by Region (Bar and Pie)
# ------------------------------------------------------------

plt.figure(figsize=(10,5))
# Bar Chart
plt.subplot(1, 2, 1)
region_sales.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales Amount ($)')
plt.xticks(rotation=45)

# Pie Chart
plt.subplot(1, 2, 2)
region_sales.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='viridis')
plt.title('Sales Distribution by Region')
plt.ylabel('')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Identify Top-Performing Regions
# ------------------------------------------------------------
top_regions = region_sales.head(3)
print("===== Top Performing Regions =====")
print(top_regions, "\n")

# ------------------------------------------------------------
# 7. Group Data by Region and Product Category
# ------------------------------------------------------------
region_category_sales = df.groupby(['Region', 'Product_Category'])['Sales_Amount'].sum().unstack()
print("===== Total Sales by Region and Product Category =====")
print(region_category_sales, "\n")

# ------------------------------------------------------------
# 8. Visualize Region vs Product Category Sales
# ------------------------------------------------------------

plt.figure(figsize=(10,6))
region_category_sales.plot(kind='bar', stacked=True, figsize=(10,6), colormap='tab20c')
plt.title('Stacked Bar Chart: Sales by Region and Product Category')
plt.xlabel('Region')
plt.ylabel('Total Sales Amount ($)')
plt.legend(title='Product Category')
plt.tight_layout()
plt.show()

# Grouped bar chart (alternative visualization)
region_category_sales.plot(kind='bar', figsize=(10,6))
plt.title('Grouped Bar Chart: Sales by Region and Product Category')
plt.xlabel('Region')
plt.ylabel('Total Sales Amount ($)')
plt.legend(title='Product Category')
plt.tight_layout()
plt.show()

print("✅ Data Aggregation and Visualization Completed Successfully!")