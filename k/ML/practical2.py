# 🚕 Predict Uber Ride Price using Linear, Ridge, and Lasso Regression

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# Step 2: Load dataset directly from URL
url = "C:\\Users\\Rutuja\\Desktop\\ML&DMV\\uber_dataset.csv"
df = pd.read_csv(url)


# Step 3: Display first few rows
print("Dataset Preview:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

# Step 4: Data Preprocessing
# Drop missing values
df.dropna(inplace=True)

# Convert datetime columns
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# Extract useful time features
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month

# Calculate distance using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)*2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)*2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in km
    return c * r

df['distance_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                              df['dropoff_latitude'], df['dropoff_longitude'])

# Step 5: Identify outliers (using boxplot)
plt.figure(figsize=(6,4))
sns.boxplot(x=df['fare_amount'])
plt.title("Outlier Detection - Fare Amount")
plt.show()

# Remove unrealistic fares and distances
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 100)]
df = df[df['distance_km'] < 50]

# Step 6: Check correlation
corr = df[['fare_amount', 'distance_km', 'hour', 'day', 'month']].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 7: Feature selection
X = df[['distance_km', 'hour', 'day', 'month']]
y = df['fare_amount']

# Step 8: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train models
lin_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)
lasso_reg = Lasso(alpha=0.1)

lin_reg.fit(X_train, y_train)
ridge_reg.fit(X_train, y_train)
lasso_reg.fit(X_train, y_train)

# Step 10: Predictions
y_pred_lin = lin_reg.predict(X_test)
y_pred_ridge = ridge_reg.predict(X_test)
y_pred_lasso = lasso_reg.predict(X_test)

# Step 11: Evaluation
def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} → R²: {r2:.3f}, RMSE: {rmse:.3f}")

print("\n📊 Model Evaluation Results:")
evaluate_model(y_test, y_pred_lin, "Linear Regression")
evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
evaluate_model(y_test, y_pred_lasso, "Lasso Regression")