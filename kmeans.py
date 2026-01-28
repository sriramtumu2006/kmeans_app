import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)

df = pd.read_csv("Wholesale customers data.csv")

numeric_features = [
    'Fresh', 'Milk', 'Grocery', 'Frozen',
    'Detergents_Paper', 'Delicassen'
]

st.sidebar.header("Clustering Controls")

feature_x = st.sidebar.selectbox("Select Feature 1", numeric_features)
feature_y = st.sidebar.selectbox("Select Feature 2", numeric_features, index=1)

k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
random_state = st.sidebar.number_input("Random State", value=42)

run = st.sidebar.button("🟦 Run Clustering")

if run:

    X = df[[feature_x, feature_y]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    st.subheader("📊 Cluster Visualization")

    plt.figure()
    plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=clusters
    )
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker='X',
        s=200
    )
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title("Customer Segments")
    st.pyplot(plt)

    st.subheader("📋 Cluster Summary")

    summary = (
        df.groupby("Cluster")[[feature_x, feature_y]]
        .agg(['count', 'mean'])
    )

    st.dataframe(summary)

    st.subheader("💡 Business Interpretation")

    for cluster_id in sorted(df['Cluster'].unique()):
        avg_x = df[df['Cluster'] == cluster_id][feature_x].mean()
        avg_y = df[df['Cluster'] == cluster_id][feature_y].mean()

        if avg_x > df[feature_x].mean() and avg_y > df[feature_y].mean():
            st.success(
                f"🟢 Cluster {cluster_id}: High-spending customers across selected categories."
            )
        elif avg_x < df[feature_x].mean() and avg_y < df[feature_y].mean():
            st.warning(
                f"🟡 Cluster {cluster_id}: Budget-conscious customers with lower annual spending."
            )
        else:
            st.info(
                f"🔵 Cluster {cluster_id}: Moderate spenders with selective purchasing behavior."
            )

    st.info(
        "📌 Customers grouped in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted using similar business strategies."
    )
