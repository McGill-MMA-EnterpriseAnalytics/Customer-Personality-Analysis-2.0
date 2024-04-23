#!/usr/bin/env python
# coding: utf-8

# # Import Library & Packages

# In[55]:


get_ipython().system('jupyter nbconvert --to script Clustering(1).ipynb')


# In[56]:


#!pip install --upgrade notebook nbconvert

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from yellowbrick.cluster import KElbowVisualizer
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


# ## Data Import

# dataset URL: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
# 
# Clustering model directly use Prepared Preprocessed Data

# In[57]:


# !wget -o ./raw_data/marketing_campaign.csv 'https://raw.githubusercontent.com/McGill-MMA-EnterpriseAnalytics/Customer-Personality-Analysis/main/Data/Prepared%20Data/Preprocessed%20Data.csv?token=GHSAT0AAAAAACL3VCJBWRLK4EZNMQNWACFWZO2XH5A'


# In[58]:


path = "https://raw.githubusercontent.com/McGill-MMA-EnterpriseAnalytics/Customer-Personality-Analysis/main/Data/Prepared%20Data/Preprocessed%20Data.csv?token=GHSAT0AAAAAACL3VCJBWRLK4EZNMQNWACFWZO2XH5A"
data = pd.read_csv(path)


# In[59]:


# path = './raw_data/marketing_campaign.csv'
# data = pd.read_csv(path, delimiter='\t')
print(data.head())


# In[60]:


data.describe()


# In[61]:


data.info()


# **Drop Total_purchase to prevent data redundancy in the clustering model**

# In[62]:


data1 = data.drop(['Total_purchase'], axis=1)
data1.info()


# ## PCA Dimention Reduction

# In[63]:


# PCA for dimensionality reduction
pca = PCA(n_components=3, random_state=42)
pca_ds = pd.DataFrame(pca.fit_transform(data1), columns=["col1", "col2", "col3"])


# ## Select the number of clusters using the elbow method

# In[64]:


# Clustering with KMean Clustering and visualizing clusters
elbow_m = KElbowVisualizer(KMeans(), k=10)
elbow_m.fit(pca_ds)
elbow_m.show()


# ## Adapt Kmean's number of clusters --> n = 4
# 
# Since we want to compare the clustering models with kmeans model, we are adapting n = 4 from Kmeans.

# In[65]:


# Initiating the Agglomerative Clustering model
ac = AgglomerativeClustering(n_clusters=4)
yhat_ac = ac.fit_predict(pca_ds)
ac_score = silhouette_score(pca_ds, yhat_ac)
print("Silhouette Score For Agglomerative Clustering", ac_score)


# In[66]:


kmeans = KMeans(n_clusters=4)
yhat_kmeans = kmeans.fit_predict(pca_ds)
k_score = silhouette_score(pca_ds, yhat_kmeans)
print("Silhouette Score For K-means Clustering:", k_score)


# In[67]:


dbscan = DBSCAN(eps=0.5, min_samples=5)  

# Fit the model to the data and predict cluster labels
yhat_dbscan = dbscan.fit_predict(pca_ds)

# Check the unique cluster labels assigned by DBSCAN
unique_labels = np.unique(yhat_dbscan)

# give warning if there is only one cluster label
if len(unique_labels) > 1:
    # Calculate the silhouette score for DBSCAN clustering
    dbscan_score = silhouette_score(pca_ds, yhat_dbscan)
    print("Silhouette Score For DBSCAN Clustering:", dbscan_score)
else:
    print("DBSCAN clustering did not assign more than one cluster label.")


# In[68]:


gmm = GaussianMixture(n_components=4)

# Fit the model to the data and predict cluster labels
yhat_gmm = gmm.fit_predict(pca_ds)

# Calculate the silhouette score for GMM clustering
gmm_score = silhouette_score(pca_ds, yhat_gmm)
print("Silhouette Score For Gaussian Mixture Model Clustering:", gmm_score)


# In[69]:


def plot_and_highlight_max(a, b, c):
    heights = [a, b, c]
    
    # Find the maximum value among a, b, c
    max_value = max(a, b, c)
    max_index = heights.index(max_value)  # Index for the highest value
    
    # Colors for the bars
    colors = ['skyblue'] * len(heights)  # Default color for all bars
    colors[max_index] = 'salmon'  # Color the maximum value bar as salmon
    
    labels = ['Agglomerative', 'Kmeans', 'GMM']
    
    # Plot the bar graph with implicit x-values and custom labels
    plt.figure(figsize=(5, 3))
    plt.bar(labels, heights, color=colors)
    plt.xticks(range(len(heights)), labels)
    
    # Set labels and title
    plt.ylabel('Silhouette Score')
    plt.title('Comparison of Silhouette Scores for Different Clustering Methods')

    
    # Show the plot
    plt.show()

plot_and_highlight_max(ac_score, k_score, gmm_score)


# ## Plot Clusters

# In[70]:


pca_ds["Clusters"] = yhat_kmeans
data["Clusters"] = yhat_kmeans
data1["Clusters"] = yhat_kmeans

# Plotting the clusters
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection='3d')
ax.scatter(pca_ds["col1"], pca_ds["col2"], pca_ds["col3"], s=40, c=pca_ds["Clusters"], cmap='RdBu')
ax.set_title("The plot of the clusters")
plt.show()


# In[71]:


# Distribution of the clusters
sns.countplot(x="Clusters", data=data, palette="RdBu")
plt.title("Distribution of the clusters")
plt.show()


# In[72]:


cluster_counts = data['Clusters'].value_counts()
labels = cluster_counts.index
sizes = cluster_counts.values

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#caced3', '#6a0523', '#173557', '#8a7786'], wedgeprops={'edgecolor': 'white'})

# Draw a white circle at the center to create a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')
plt.title("Distribution of the clusters")
plt.show()


# In[73]:


# Profile of a group of individuals based on their income and spending habits
sns.scatterplot(data=data, x="Total_purchase", y="Income", hue="Clusters", palette="RdBu")
plt.title("Profile of a group of individuals based on their income and spending habits")
plt.legend()
plt.show()


# In[74]:


# Plotting count of total campaign accepted
plt.figure()
fig = sns.countplot(data=data, x="Cmp_Attitude", hue="Clusters", palette="RdBu")
fig.set_title("Count of promotion accepted")
fig.set_xlabel("Number of total accepted promotions")
#fig.set_xticklabels([])  # Hide x-axis labels
plt.show()


# In[75]:


# Plotting the number of deals purchased
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="NumDealsPurchases", palette="RdBu")
pl.set_title("Number of deals purchased")
plt.show()


# In[76]:


# Plotting the complain
plt.figure()
pl = sns.countplot(data=data, x="Complain", hue="Clusters", palette="RdBu")
pl.set_title("Count of Complained Customers")
plt.show()


# In[77]:


plt.figure()
complain_counts = data['Complain'].value_counts()
plt.pie(complain_counts, labels=complain_counts.index, autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'])
plt.title("Count of Complained Customers")
plt.gca().add_artist(plt.Circle((0,0),0.70,fc='white'))
plt.axis('equal')
plt.show()


# In[78]:


# Plotting the Response
plt.figure()
pl = sns.countplot(data=data, x="Response", hue="Clusters", palette="RdBu")
pl.set_title("Count of Response")
plt.show()


# In[79]:


# Plotting the marriage
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="Total_purchase", palette="RdBu")
pl.set_title("Total Purchase")
plt.show()


# In[80]:


cluster_iqr = data.groupby('Clusters')['Total_purchase'].describe()[['25%', '75%']]
print(cluster_iqr)


# In[81]:


# Plotting the marriage
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="Family_Size", palette="RdBu")
pl.set_title("Family Size")
plt.show()


# In[82]:


# Plotting the income
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="Income", palette="RdBu")
pl.set_title("Income")
plt.show()


# In[83]:


cluster_iqr = data.groupby('Clusters')['Income'].describe()[['25%', '75%']]
print(cluster_iqr)


# In[84]:


# Plotting the Age
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="Age", palette="RdBu")
pl.set_title("Age")
plt.show()


# In[85]:


cluster_iqr = data.groupby('Clusters')['Age'].describe()[['25%', '75%']]
print(cluster_iqr)


# In[86]:


# Plotting the Member_Year
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="Member_Year", palette="RdBu")
pl.set_title("Membership Year")
plt.show()


# In[87]:


# Plotting the Recency
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="Recency", palette="RdBu")
pl.set_title("Recency")
plt.show()


# In[88]:


# Plotting the Marital_Status_Married
plt.figure()
pl = sns.countplot(data=data, x="Marital_Status_Married", hue="Clusters", palette="RdBu")
pl.set_title("Count of Married Customer")
plt.show()


# In[89]:


# Plotting the KidHome
plt.figure()
pl = sns.countplot(data=data, x="Is_Parent", hue="Clusters", palette="RdBu")
pl.set_title("Count of Kid Home")
plt.show()


# In[90]:


# Plotting the Teenhome
plt.figure()
pl = sns.countplot(data=data, x="Family_Size", hue="Clusters", palette="RdBu")
pl.set_title("Count of Family Size")
plt.show()


# In[91]:


# Plotting the MntWines
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="MntWines", palette="RdBu")
pl.set_title("Amount Wines Purchased")
plt.show()


# In[92]:


# Plotting the MntFruits
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="MntFruits", palette="RdBu")
pl.set_title("Amount Fruits Purchased")
plt.show()


# In[93]:


# Plotting the MntMeatProducts
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="MntMeatProducts", palette="RdBu")
pl.set_title("Amount Meat Products Purchased")
plt.show()


# In[94]:


# Plotting the MntFishProducts
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="MntFishProducts", palette="RdBu")
pl.set_title("Amount Fish Purchased")
plt.show()


# In[95]:


# Plotting the MntSweetProducts
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="MntSweetProducts", palette="RdBu")
pl.set_title("Amount Sweet Purchased")
plt.show()


# In[96]:


# Plotting the MntGoldProds
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="MntGoldProds", palette="RdBu")
pl.set_title("Amount Gold Products Purchased")
plt.show()


# In[97]:


# Plotting the NumWebPurchases
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="NumWebPurchases", palette="RdBu")
pl.set_title("Number of Web Purchases")
plt.show()


# In[98]:


# Plotting the NumStorePurchases
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="NumStorePurchases", palette="RdBu")
pl.set_title("Number of Store Purchases")
plt.show()


# In[99]:


data["PercentWebpurchases"]= (data["NumWebPurchases"] / (data["NumWebPurchases"] + data["NumStorePurchases"])) * 100


# In[100]:


# Plotting the PercentWebpurchases
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="PercentWebpurchases", palette="RdBu")
pl.set_title("Percentage of Web Perchases")
plt.show()


# In[101]:


# Plotting the NumCatalogPurchases
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="NumCatalogPurchases", palette="RdBu")
pl.set_title("Number of Category Purchases")
plt.show()


# In[102]:


# Plotting the NumWebVisitsMonth
plt.figure()
pl = sns.boxenplot(data=data, x="Clusters", y="NumWebVisitsMonth", palette="RdBu")
pl.set_title("Number of Web Visits")
plt.show()


# # Export Cluster data

# In[103]:


data.to_csv('./preprocessed_data/Clustered_with_Total_purchase.csv', index=False)


# In[104]:


data1.to_csv('./preprocessed_data/Clustered_without_Total_purchase.csv', index=False)


# In[105]:


data.head(10)


# In[106]:


data1.head(10)


# In[107]:


data = pd.read_csv(path)

# Drop Total_purchase to prevent data redundancy in the clustering model
data1 = data.drop(['Total_purchase'], axis=1)

# PCA for dimensionality reduction
pca = PCA(n_components=3, random_state=42)
pca_ds = pd.DataFrame(pca.fit_transform(data1), columns=["col1", "col2", "col3"])

# Select the number of clusters using the elbow method
elbow_m = KElbowVisualizer(KMeans(), k=10)
elbow_m.fit(pca_ds)
elbow_m.show()

# Initialize clustering models
ac = AgglomerativeClustering(n_clusters=4)
kmeans = KMeans(n_clusters=4)
dbscan = DBSCAN(eps=0.5, min_samples=5)
gmm = GaussianMixture(n_components=4)

# Fit models and calculate silhouette scores
models = [ac, kmeans, dbscan, gmm]
scores = []

for model in models:
    if model == dbscan:
        # Special case for DBSCAN
        labels = model.fit_predict(pca_ds)
        if len(set(labels)) > 1:  # Check if DBSCAN found more than one cluster
            score = silhouette_score(pca_ds, labels)
            scores.append(score)
        else:
            scores.append(-1)  # DBSCAN did not find meaningful clusters
    else:
        labels = model.fit_predict(pca_ds)
        score = silhouette_score(pca_ds, labels)
        scores.append(score)

# Select the best model based on silhouette score
best_score_index = scores.index(max(scores))
best_model = models[best_score_index]

# Apply the best model to the data
best_labels = best_model.fit_predict(pca_ds)
data['Cluster'] = best_labels

# Ensemble Clustering: Assuming we have multiple clustering results to ensemble
# Export Clustered data
data.to_csv('./preprocessed_data/Clustered_with_Best_Method.csv', index=False)

# Plotting clusters and distributions as needed
sns.countplot(x="Cluster", data=data, palette="RdBu")
plt.title("Distribution of Clusters")
plt.show()

