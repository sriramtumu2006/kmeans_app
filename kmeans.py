import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Wholesale Customer Segmentation", layout="wide")

st.title("🛒 Wholesale Customers Segmentation using K-Means")

df = pd.read_csv("Wholesale customers data.csv")

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.subheader("📐 Dataset Shape")
st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

X = df[['Fresh', 'Milk', 'Grocery', 'Frozen',
        'Detergents_Paper', 'Delicassen']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.subheader("📉 Elbow Method")

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
st.pyplot(plt)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

st.subheader("📊 Cluster Scatter Plot")

plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.title("Customer Clusters")
st.pyplot(plt)

st.subheader("🧩 Clustered Data")
st.dataframe(df.head(10))
st.write("Cluster Counts:")
st.write(df['Cluster'].value_counts())
st.markdown("""
### 📌 Conclusion
The K-Means clustering algorithm effectively segments wholesale customers into distinct groups based on their purchasing behavior. This segmentation can help businesses tailor their marketing strategies and improve customer retention.
""")