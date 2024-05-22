#%%
import pandas as pd
import numpy as np
import os 
os.chdir("/Users/chentahung/Desktop/git/GMM-Visualization")

from src.main.GMMViz.GaussianMixtureModel import GMM
# from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

#%%

if __name__ == '__main__':
    # Load the dataset
    data = pd.DataFrame(np.random.rand(1000,3), columns = ["A", "B", "C"])
    
    nk = 3
    
    # Create an instance of the GMM class
    gmm = GMM(data, n_clusters=nk, random_state=42, max_iter = 30,  tol = 1e-4)
    gmm.fit()
    
    # data['GMMcluster'] = gmm.predict(data)
    
    # skgmm = GaussianMixture(n_components=nk, random_state=42)
    # skgmm.fit(data)
    # data['sklcluster'] = skgmm.predict(data)
        
    # print(pd.crosstab(data['GMMcluster'], data['sklcluster']))
    
    # plt.plot(gmm.estimands_logLikelihood)
    # plt.show()
#%%


if __name__ == '__main__':
    # Load the dataset
    data = pd.read_csv('data/Customer_Data.csv')
    
    data = data.drop(['CUST_ID', 'CREDIT_LIMIT', 'MINIMUM_PAYMENTS'], axis = 1) #contains Nan
    
    nk =3
    # apply scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    # X = data.values
    # Create an instance of the GMM class
    gmm = GMM(X, n_clusters=nk)
    gmm.fit()
    
    # # data['GMMcluster'] = gmm.predict(data)
    
    # skgmm = GaussianMixture(n_components=nk)
    # skgmm.fit(X)
    # # data['sklcluster'] = skgmm.predict(data)
        
    # print(pd.crosstab(gmm.predict(X), skgmm.predict(X)))
    # plt.plot(gmm.estimands_logLikelihood)
    # plt.show()
#%%
if __name__ == '__main__':
    # Load the dataset
    data = pd.read_csv('data/segmentation data.csv')
    
    data = data.drop(['ID'], axis = 1) #contains Nan
    
    nk = 3
    # apply scaling
    scaler = StandardScaler()
    datascaled = scaler.fit_transform(data)
    # X = data.values
    # Create an instance of the GMM class
    gmm = GMM(datascaled, n_clusters=nk)
    resp = gmm.fit()
    
    data['GMMcluster'] = gmm.predict(datascaled)
    
    skgmm = GaussianMixture(n_components=nk)
    skgmm.fit(datascaled)
    data['sklcluster'] = skgmm.predict(datascaled)
        
    print(pd.crosstab(data['GMMcluster'], data['sklcluster']))
    
    # Silhouette Score
    print("GMM Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(data, data['GMMcluster']))

    print("SKL Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(data, data['sklcluster']))

# %%

import matplotlib.pyplot as plt

n_clusters_range = range(2, 11)
bic_scores = []

for n in n_clusters_range:
    gmm = GMM(datascaled, n_clusters=n)
    gmm.fit()
    bic = gmm.bic()
    bic_scores.append(bic)

# Plotting the BIC scores
plt.figure(figsize=(8, 5))
plt.plot(n_clusters_range, bic_scores, marker='o')
plt.title('BIC Scores by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC Score')
plt.grid(True)

#%%

from sklearn.mixture import GaussianMixture
import numpy as np

# Assume X is your dataset and true_labels are the true cluster labels
gmm = GMM(data, n_clusters=2)
gmm.fit()
gmmlabels = gmm.predict(data)

skgmm = GaussianMixture(n_components=2)
skgmm.fit(data)
sklabels = skgmm.predict(data)

#%%
# Calinski-Harabasz Index
print("Calinski-Harabasz Index: %0.3f"
      % metrics.calinski_harabasz_score(data, labels))

# If true labels are available
# if 'true_labels' in locals():
#     print("Adjusted Rand Index: %0.3f"
#           % metrics.adjusted_rand_score(true_labels, labels))
#     print("Normalized Mutual Information: %0.3f"
#           % metrics.normalized_mutual_info_score(true_labels, labels))

