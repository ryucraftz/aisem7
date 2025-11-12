# ✅ Random Forest Classifier to Predict Car Safety

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load dataset (update path if needed)
df = pd.read_csv(r"C:\Users\Rutuja\Desktop\ML&DMV\car_evaluation.csv")

# Step 2: Rename columns properly
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Step 3: Encode categorical data
le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# Step 4: Split data
X = df.drop('class', axis=1)
y = df['class']

# Step 5: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate
print("🎯 Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: Feature importance plot
plt.figure(figsize=(8,5))
plt.barh(X.columns, model.feature_importances_, color='teal')
plt.title("Feature Importance in Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()