import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import joblib # For saving/loading models

# - 1. Load and inspect data -

df = pd.read_csv("/content/drive/MyDrive/Store Demand Forecast Weekly.csv")

print("- Initial DataFrame Head -")
print(df.head())
print("\n- Initial DataFrame Info -")
df.info()
print("\n- Initial DataFrame Description -")
print(df.describe())

# - 2. Data Cleaning and Feature Engineering -

# Convert relevant columns to numeric, coercing errors to NaN

df['sell_quantity'] = pd.to_numeric(df['sell_quantity'], errors='coerce')
df['unit_cost_amount'] = pd.to_numeric(df['unit_cost_amount'], errors='coerce')
df['final_fcst_each_qty'] = pd.to_numeric(df['final_fcst_each_qty'], errors='coerce')


df['total_sales'] = df['sell_quantity'] * df['unit_cost_amount']

# Replace NaN and Inf values with 0 across the entire DataFrame

df = df.replace([np.nan, np.inf], 0)
print("\n--- DataFrame after NaN/Inf replacement with 0 ---")
print(df.head())
print("\n--- DataFrame Info after NaN/Inf replacement ---")
df.info()


store_features = df.groupby('store_nbr').agg({
    'sell_quantity': 'mean',
    'unit_cost_amount': 'mean',
    'final_fcst_each_qty': 'mean',
    'total_sales': 'sum' # Summing total_sales per store_nbr
}).reset_index()

print("\n- Aggregated Store Features Head  -")
print(store_features.head())
print("\n- Missing values in aggregated features before final imputation -")
print(store_features.isna().sum())


# - 3. Scaling and Imputation for Clustering Features -

# Select features for scaling and clustering
# Bas
clustering_features = ['sell_quantity', 'unit_cost_amount', 'final_fcst_each_qty', 'total_sales']

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the selected features
scaled_data = scaler.fit_transform(store_features[clustering_features])

# Impute missing values (e.g., from unit_cost_amount if it was NaN before aggregation)

imputer = SimpleImputer(strategy='mean')
scaled_data_imputed = imputer.fit_transform(scaled_data)

print("\n- Missing values in scaled and imputed data (should be 0) -")
print(pd.DataFrame(scaled_data_imputed, columns=clustering_features).isna().sum())


# - 4. Dimensionality Reduction (PCA) -
# Reduce to 2 components for visualization
pca = PCA(n_components=2, random_state=42)
pca_data = pca.fit_transform(scaled_data_imputed)

# Add PCA components back to the store_features DataFrame
store_features['pca_x'] = pca_data[:, 0]
store_features['pca_y'] = pca_data[:, 1]

# - 5. K-Means Clustering -
# Initialize KMeans with 5 clusters as specified in your problem
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10) # n_init=10 is good practice for KMeans

# Fit KMeans and predict clusters
store_features['cluster'] = kmeans.fit_predict(scaled_data_imputed)

print("\n- Store Features with PCA and Cluster Labels Head -")
print(store_features.head())
print("\n- Cluster Distribution -")
print(store_features['cluster'].value_counts().sort_index())

# - 6. Save Outputs -
# Save the clustered data to a CSV file
store_features.to_csv("store_clusters.csv", index=False)
print("\n'store_clusters.csv' saved successfully.")

# Save the trained KMeans model
joblib.dump(kmeans, "kmeans_model.pkl")
print("'kmeans_model.pkl' saved successfully.")

# --- 7. Visualize Clusters (Matplotlib/Seaborn) ---

# Scatter plot of clusters based on PCA components
plt.figure(figsize=(10, 6))
sns.scatterplot(data=store_features, x='pca_x', y='pca_y', hue='cluster', palette='Set2', s=100)
plt.title("Store Clusters (KMeans + PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("cluster_plot.png") # Save the plot as an image
plt.show()

# --- 8. Visualize Cluster Characteristics (Box Plots) ---
# This helps understand what defines each cluster based on original features
features_to_plot = ['sell_quantity', 'final_fcst_each_qty', 'total_sales'] # From your PDF

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y=feature, data=store_features)
    plt.title(f'Distribution of {feature} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- 9. Visualize Cluster Characteristics (Heatmap of Mean Values) ---
# Calculate the mean of the features for each cluster
cluster_means = store_features.groupby('cluster')[features_to_plot].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Mean Feature Values by Cluster')
plt.xlabel('Features')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()

print("\n- Analysis Complete -")
