import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os 

st.set_page_config(layout="wide") 
st.title("Store Clustering Dashboard")

if os.path.exists("store_clusters.csv"):
    df = pd.read_csv("store_clusters.csv")

    st.write("### Clustered Store Data")
    st.dataframe(df)

    st.write("### Cluster Plot")

    #scatter plot using seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='pca_x', y='pca_y', hue='cluster', s=100, palette='Set2', ax=ax)
    ax.set_title("Store Clusters (PCA Projection)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    st.pyplot(fig)

