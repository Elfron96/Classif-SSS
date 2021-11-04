#%% Side Scan Sonar preprocessing


#%% Import some packages

import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from roipoly import roipoly 
from displayImageLabel import displayImageLabel
from cptPatchesStats import cptPatchesStats
from scipy.io import loadmat
from kneed import *
plt.ioff()


#%% Partie 1

# Chargement et affichage de l'image à analyser

data = loadmat("12_data.mat")

# Affichage de l'image
data_img = data['img']


plt.figure()
plt.title("Affichage de l'image'")
plt.imshow(data_img, aspect = "auto")
plt.xlabel("Echantillon")
plt.ylabel("N° de Pings")
plt.colorbar(label ="Backscatter [dB]")




#%% classif non_supervisée

N = data_img.reshape((data_img.size, 1))# Size prend le nombre d'éléments dans la matrice (à ne pas confondre avec shape)
Nech = N[::50]

## Test sur un sous échantillon 
# Application du K-Means : 
model_kmeans = KMeans(n_clusters = 3, n_init=20)
model_kmeans.fit(Nech)

# kmeans_kwargs = {
# "init": "random",
# "n_init": 20,
# }

# # A list holds the SSE values for each k
# sse = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(Nech)
#     sse.append(kmeans.inertia_)
# plt.figure()
# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
# print(kl.elbow)





# #Prédictions des classes : 
Yech = model_kmeans.predict(Nech)

# # Affichage des résultats : 
plt.figure()
plt.scatter(Nech, Yech, c= Yech, label = "Classes")
plt.title("Classes déterminées par K-Means")
plt.xlabel("Pixels")
plt.ylabel("Classes")

# ## K-Means sur toute l'image : 
# # On garde l'a priori avec 3 classes : 
model_kmeans.fit(N)
Y = model_kmeans.predict(N)


plt.figure()
label_plot, plot =  displayImageLabel(Y, data_img)

# plt.plot(Y,label=label_plot)




#%% On enleve colonne d'eau et classification

data_img = data_img[:,320:]
print(data_img.shape)
N = data_img.reshape((data_img.size, 1))# Size prend le nombre d'éléments dans la matrice (à ne pas confondre avec shape)
Nech = N[::50]
Yech = model_kmeans.predict(Nech)
model_kmeans = KMeans(n_clusters = 3, n_init=20)
model_kmeans.fit(Nech)

plt.figure()
plt.scatter(Nech, Yech, c= Yech, label = "Classes")
plt.title("Classes déterminées par K-Means")
plt.xlabel("Pixels")
plt.ylabel("Classes")

# ## K-Means sur toute l'image : 
# # On garde l'a priori avec 3 classes : 
model_kmeans.fit(N)
Y = model_kmeans.predict(N)


plt.figure()
label_plot, plot =  displayImageLabel(Y, data_img)
#%% Correction de l'image par moyennage simple et affichage

# correction


# classification
    






#%% Classification supervisée


# Affichage de l'image corrigée


# segmentation manuelle




#%% Affichage des histogrammes et histo cumulé



#%% Calcul des descripteurs stats


## calcul des features


# classification par kmeans








plt.show()