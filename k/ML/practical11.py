# ------------------------------------------------------------
# Project: Data Visualization using Matplotlib - Air Quality Index Analysis
# ------------------------------------------------------------

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Import Dataset
# ------------------------------------------------------------

file_path = "City_Air_Quality.csv"

try:
    df = pd.read_csv(file_path)
except:
    # Create sample dataset if file not found
    data = {
        'Date': pd.date_range(start='2024-01-01', periods=15, freq='D'),
        'PM2.5': [120, 115, 98, 87, 130, 110, 92, 105, 99, 85, 88, 95, 102, 112, 90],
        'PM10': [160, 155, 140, 132, 170, 158, 145, 138, 140, 130, 125, 150, 142, 148, 135],
        'CO': [0.9, 0.8, 1.0, 0.85, 1.1, 1.0, 0.95, 0.9, 1.05, 0.8, 0.75, 0.92, 0.88, 1.0, 0.84],
        'AQI': [180, 170, 160, 155, 200, 190, 170, 160, 150, 140, 135, 155, 165, 175, 145]
    }
    df = pd.DataFrame(data)

print("===== Dataset Preview =====")
print(df.head(), "\n")

# ------------------------------------------------------------
# 2. Explore Dataset
# ------------------------------------------------------------
print("Dataset Info:")
print(df.info(), "\n")

print("Missing Values:\n", df.isnull().sum(), "\n")

# Convert date column to datetime (if not already)
df['Date'] = pd.to_datetime(df['Date'])

# ------------------------------------------------------------
# 3. Identify Relevant Variables
# ------------------------------------------------------------
pollutants = ['PM2.5', 'PM10', 'CO']
aqi = 'AQI'

# ------------------------------------------------------------
# 4. Line Plot - AQI Trend Over Time
# ------------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['AQI'], color='crimson', marker='o', linestyle='-', linewidth=2)
plt.title('Overall AQI Trend Over Time', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Air Quality Index (AQI)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. Line Plots - Pollutant Trends Over Time
# ------------------------------------------------------------
plt.figure(figsize=(10,6))
for p in pollutants:
    plt.plot(df['Date'], df[p], marker='o', label=p)
plt.title('Pollutant Levels Over Time', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Pollutant Concentration')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Bar Plot - AQI by Date
# ------------------------------------------------------------
plt.figure(figsize=(10,5))
plt.bar(df['Date'].dt.strftime('%Y-%m-%d'), df['AQI'], color='skyblue', edgecolor='black')
plt.xticks(rotation=45)
plt.title('AQI Comparison Across Dates')
plt.xlabel('Date')
plt.ylabel('AQI Value')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Box Plot - AQI Distribution by Pollutant
# ------------------------------------------------------------
# Melt data for easy plotting
melted = pd.melt(df, id_vars=['Date', 'AQI'], value_vars=pollutants, var_name='Pollutant', value_name='Concentration')

plt.figure(figsize=(8,5))
sns.boxplot(x='Pollutant', y='AQI', data=melted, palette='viridis')
plt.title('Distribution of AQI by Pollutant Category')
plt.ylabel('AQI')
plt.xlabel('Pollutant')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. Scatter Plot - Relationship Between Pollutants and AQI
# ------------------------------------------------------------
plt.figure(figsize=(12,4))
for i, p in enumerate(pollutants, 1):
    plt.subplot(1, 3, i)
    plt.scatter(df[p], df['AQI'], color='orange', edgecolor='black')
    plt.title(f"{p} vs AQI")
    plt.xlabel(p)
    plt.ylabel('AQI')
    plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. Bubble Chart - Multi-Variable Visualization
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['PM2.5'], df['PM10'], s=df['AQI'], alpha=0.5, c=df['AQI'], cmap='coolwarm', edgecolors='black')
plt.title('Bubble Chart: PM2.5 vs PM10 (Bubble size = AQI)')
plt.xlabel('PM2.5 Level')
plt.ylabel('PM10 Level')
plt.colorbar(label='AQI')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. Correlation Heatmap
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(df[['PM2.5','PM10','CO','AQI']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Pollutants and AQI')
plt.tight_layout()
plt.show()

print("✅ Visualization completed successfully!")