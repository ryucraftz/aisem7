# --------------------------------------------------------
# Project: Analyzing Sales Data from Multiple File Formats
# --------------------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. Load Sales Data from Multiple File Formats
# --------------------------------------------------------

# Example file paths (replace with your actual dataset paths)
csv_file = "sales_data.csv"
excel_file = "sales_data.xlsx"
json_file = "sales_data.json"

# If files not found, we’ll create dummy data
try:
    sales_csv = pd.read_csv(csv_file)
    sales_excel = pd.read_excel(excel_file)
    sales_json = pd.read_json(json_file)
except:
    # Create sample dataset
    data = {
        'OrderID': [101, 102, 103, 104, 105],
        'Product': ['A', 'B', 'C', 'A', 'B'],
        'Quantity': [2, 1, 3, 4, 2],
        'Price': [200, 150, 100, 200, 150],
        'Customer': ['John', 'Alice', 'Mark', 'John', 'Eve']
    }

    # Simulate different file types
    sales_csv = pd.DataFrame(data)
    sales_excel = sales_csv.copy()
    sales_json = sales_csv.copy()

# --------------------------------------------------------
# 2. Explore Data Structure and Check for Issues
# --------------------------------------------------------

print("===== CSV Data Preview =====")
print(sales_csv.head(), "\n")

print("===== Excel Data Info =====")
print(sales_excel.info(), "\n")

print("===== JSON Data Description =====")
print(sales_json.describe(), "\n")

# Introduce missing values and duplicates for cleaning demo
sales_csv.loc[2, 'Product'] = np.nan
sales_csv = pd.concat([sales_csv, sales_csv.iloc[1:2]])  # duplicate row

# --------------------------------------------------------
# 3. Data Cleaning
# --------------------------------------------------------

print("===== Before Cleaning =====")
print(sales_csv, "\n")

# Fill missing values
sales_csv['Product'].fillna('Unknown', inplace=True)

# Remove duplicates
sales_csv.drop_duplicates(inplace=True)

# Remove invalid rows (negative or zero quantities)
sales_csv = sales_csv[sales_csv['Quantity'] > 0]

print("===== After Cleaning =====")
print(sales_csv, "\n")

# --------------------------------------------------------
# 4. Combine into a Unified DataFrame
# --------------------------------------------------------

unified_sales = pd.concat([sales_csv, sales_excel, sales_json], ignore_index=True)
print("===== Unified Combined Data =====")
print(unified_sales.head(), "\n")

# --------------------------------------------------------
# 5. Data Transformation
# --------------------------------------------------------

# Derive Total_Sale and Order_Type
unified_sales['Total_Sale'] = unified_sales['Quantity'] * unified_sales['Price']
unified_sales['Order_Type'] = np.where(unified_sales['Quantity'] > 2, 'Bulk', 'Regular')

print("===== Transformed Data =====")
print(unified_sales.head(), "\n")

# --------------------------------------------------------
# 6. Data Analysis
# --------------------------------------------------------

print("===== Descriptive Statistics =====")
print(unified_sales.describe(), "\n")

# Aggregate total sales by product
sales_by_product = unified_sales.groupby('Product')['Total_Sale'].sum()
avg_order_value = unified_sales['Total_Sale'].mean()
product_distribution = unified_sales['Product'].value_counts()

print("===== Total Sales by Product =====")
print(sales_by_product, "\n")

print("Average Order Value:", avg_order_value)
print("\n===== Product Distribution =====")
print(product_distribution, "\n")

# --------------------------------------------------------
# 7. Visualization
# --------------------------------------------------------

plt.figure(figsize=(12, 5))

# Bar Plot - Total Sales by Product
plt.subplot(1, 3, 1)
sales_by_product.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total Sales (₹)')

# Pie Chart - Product Distribution
plt.subplot(1, 3, 2)
product_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#6baed6','#fd8d3c','#74c476','#9e9ac8'])
plt.title('Product Category Distribution')
plt.ylabel('')

# Box Plot - Total Sales Distribution
plt.subplot(1, 3, 3)
plt.boxplot(unified_sales['Total_Sale'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Sales Value Distribution')

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 8. Key Insights
# --------------------------------------------------------
print("===== Key Insights =====")
print(f"1️⃣ Highest selling product: {sales_by_product.idxmax()}")
print(f"2️⃣ Lowest selling product: {sales_by_product.idxmin()}")
print(f"3️⃣ Average Order Value: ₹{round(avg_order_value,2)}")
print("4️⃣ Product distribution shows top-performing categories.\n")

print("✅ Analysis complete!")