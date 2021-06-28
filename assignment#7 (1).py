import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
 
def PCA(X , num_components):
    X_meaned = X - np.mean(X , axis = 0)
    cov_mat = np.cov(X_meaned , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
    return X_reduced

def K_Means_cluster(X, num_clusters):
    global centroids
    # setting the number of training examples
    m=X.shape[0]
    n=X.shape[1] 
    n_iter=50
    K=num_clusters
    # creating an empty centroid array
    centroids=np.array([]).reshape(n,0) 
    # creating k random centroids
    for k in range(K):
        centroids=np.c_[centroids,X[random.randint(0,m-1)]]
  
    for i in range(n_iter):
          euclid=np.array([]).reshape(m,0)
          for k in range(K):
              dist=np.sum((X-centroids[:,k])**2,axis=1)
              euclid=np.c_[euclid,dist]
          C=np.argmin(euclid,axis=1)+1
          cent={}
          for k in range(K):
               cent[k+1]=np.array([]).reshape(2,0)
          for k in range(m):
               if n ==1:  #to check 1-dimension (one component PCA)
                   cent[C[k]]=np.concatenate((cent[C[k]], X[k]), axis=None)
                   #cent[C[k]]=np.c_[([cent[C[k]],X[k]], axis=None)]
               else:
                   cent[C[k]]=np.c_[cent[C[k]],X[k]]
               #print(cent)
          for k in range(K):
               cent[k+1]=cent[k+1].T
          for k in range(K):
               centroids[:,k]=np.mean(cent[k+1],axis=0)
          final=cent
          
    return final
          
        
def plot(final, K):
    for i in range(K):
        if len(final[1].shape) ==1: #for check 1-dimension (one component PCA)
            plt.scatter(final[i+1], np.zeros_like(final[i+1]))
        else:
            plt.scatter(final[i+1][:,0],final[i+1][:,1])
    if len(final[1].shape) ==1: #for check 1-dimension (one component PCA)
        plt.scatter(centroids[0,:],np.zeros_like(centroids[0,:]),s=300,c='yellow')
    else:
        plt.scatter(centroids[0,:],centroids[1,:],s=300,c='yellow')
    plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
    #plt.savefig('one_component.jpg')
    #plt.savefig('two_components.jpg')
    #plt.savefig('data_before_pca.jpg')
    plt.show()

def load_data(path):
    data= pd.read_excel(path)
    data= data.drop('Unnamed: 0',1)
    data= data.drop('Unnamed: 3',1)
    data= data.drop([272],0)
    #data = data - data.mean()
    return data.values

if __name__ == "__main__":
    path="faithful.xlsx"
    X=load_data(path)
    #print(data.shape)
    #Applying PCA function for one component
    matrix1_reduced = PCA(X , 1)
    #Applying PCA function for two components
    matrix2_reduced = PCA(X , 2)
    plot(K_Means_cluster(matrix1_reduced, 4), 4) #plot for one component of PCA with 4 clusters
    plot(K_Means_cluster(matrix2_reduced, 4), 4)  #plot for two components of PCA with 4 clusters
    plot(K_Means_cluster(X, 4), 4) #plot for data before PCA with 4 clusters
    
    #print(len(K_Means_cluster(matrix1_reduced, 4)[1].shape))
    
    
    
    