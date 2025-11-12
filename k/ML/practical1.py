# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
url = "https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv"
df = pd.read_csv(url)

# Display first few rows
print("Dataset Preview:\n", df.head())

# Step 2: Separate features and target variable
X = df.drop('Customer_Segment', axis=1)
y = df['Customer_Segment']

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA
pca = PCA(n_components=2)   # Reduce to 2 principal components
X_pca = pca.fit_transform(X_scaled)

# Step 5: Create a new DataFrame with principal components
pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Customer_Segment'] = y

# Step 6: Visualize the PCA result
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Principal Component 1',
    y='Principal Component 2',
    hue='Customer_Segment',
    data=pca_df,
    palette='Set1',
    s=80
)
plt.title('PCA on Wine Dataset (2 Principal Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Wine Type')
plt.show()

# Step 7: Explained variance ratio
print("\nExplained variance ratio by each principal component:")
print(pca.explained_variance_ratio_)

# Step 8: Cumulative variance explained
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print("\nCumulative variance explained by 2 components: {:.2f}%".format(cumulative_variance[-1]*100))
