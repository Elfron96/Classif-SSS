#%% Side Scan Sonar preprocessing


#%% Import some packages

import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from roipoly import roipoly

from displayImageLabel import displayImageLabel
from cptPatchesStats import cptPatchesStats
from scipy.io import loadmat
from kneed import *

from selectClassArea import selectClassArea
plt.ioff()

image_plot = 0

#%% Partie 1

# Chargement et affichage de l'image à analyser

data = loadmat("12_data.mat")

# Affichage de l'image
data_img = data['img']

# if image_plot:
#     plt.figure()
#     plt.title("Affichage de l'image'")
#     plt.imshow(data_img, aspect = "auto")
#     plt.xlabel("Echantillon")
#     plt.ylabel("N° de Pings")
#     plt.colorbar(label ="Backscatter [dB]")




# #%% classif non_supervisée

# N = data_img.reshape((data_img.size, 1))# Size prend le nombre d'éléments dans la matrice (à ne pas confondre avec shape)
# Nech = N[::50]

# ## Test sur un sous échantillon 
# # Application du K-Means : 
# model_kmeans = KMeans(n_clusters = 3, n_init=20)
# model_kmeans.fit(Nech)

# kmeans_kwargs = {
# "init": "random",
# "n_init": 20,
# }

# # # A list holds the SSE values for each k
# sse = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(Nech)
#     sse.append(kmeans.inertia_)

# if image_plot:
#     plt.figure()
#     plt.style.use("fivethirtyeight")
#     plt.plot(range(1, 11), sse)
#     plt.xticks(range(1, 11))
#     plt.xlabel("Number of Clusters")
#     plt.ylabel("SSE")
#     kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
#     print(kl.elbow)





# # #Prédictions des classes : 
# Yech = model_kmeans.predict(Nech)

# # # Affichage des résultats : 
# if image_plot:
#     plt.figure()
#     plt.scatter(Nech, Yech, c= Yech, label = "Classes")
#     plt.title("Classes déterminées par K-Means")
#     plt.xlabel("Pixels")
#     plt.ylabel("Classes")

# # ## K-Means sur toute l'image : 
# # # On garde l'a priori avec 3 classes : 
# model_kmeans.fit(N)
# Y = model_kmeans.predict(N)

# if image_plot:
#     plt.figure()
#     label_plot, plot =  displayImageLabel(Y, data_img)

#     plt.plot(Y,label=label_plot)




# #%% On enleve colonne d'eau et classification

data_img = data_img[:,320:]
# # print(data_img.shape)
# N = data_img.reshape((data_img.size, 1))# Size prend le nombre d'éléments dans la matrice (à ne pas confondre avec shape)
# Nech = N[::50]
# Yech = model_kmeans.predict(Nech)
# model_kmeans = KMeans(n_clusters = 3, n_init=20)
# model_kmeans.fit(Nech)


# if image_plot:
#     plt.figure()
#     plt.scatter(Nech, Yech, c= Yech, label = "Classes")
#     plt.title("Classes déterminées par K-Means")
#     plt.xlabel("Pixels")
#     plt.ylabel("Classes")

# # ## K-Means sur toute l'image : 
# # # On garde l'a priori avec 3 classes : 
# model_kmeans.fit(N)
# Y = model_kmeans.predict(N)

# if image_plot:
#     plt.figure()
#     label_plot, plot =  displayImageLabel(Y, data_img)
# #%% Correction de l'image par moyennage simple et affichage

moy = np.mean(data_img, axis = 0)

# # correction
img_corrige_moy = data_img / moy
# if image_plot:
#     plt.figure()

#     plt.title("Affichage de l'image initiale")
#     plt.imshow(data_img, aspect = "auto")
#     plt.xlabel("Échantillon")
#     plt.ylabel("Pings")
#     plt.colorbar(label ="BS [dB]")

#     plt.figure()
#     plt.plot(moy)
#     plt.xlabel("Échantillon")
#     plt.ylabel("Moyenne sur l'ensemble des pings")
#     plt.title("Evolution de la moyenne sur l'ensemble des pings en fonction du numéro d'échantillon")

#     plt.figure()
#     plt.imshow(img_corrige_moy)
#     plt.title("Image corrigée de la moyenne")
#     plt.xlabel("Échantillon")
#     plt.ylabel("Moyenne sur l'ensemble des pings")
#     plt.colorbar(label =  "Backscatter [dB]")

#     plt.figure()
#     plt.plot(np.mean(img_corrige_moy, axis = 0))
#     plt.xlabel("Numéro d'échantillon")
#     plt.ylabel("Moyenne sur l'ensemble des pings")
#     plt.title("Evolution de la moyenne sur l'ensemble des pings en fonction du numéro d'échantillon sur l'image corrigé")


# # classification
    
# N_corr_moyenne = img_corrige_moy.reshape((img_corrige_moy.size, 1))

# model_kmeans.fit(N_corr_moyenne)
# Y_corr_moy = model_kmeans.predict(N_corr_moyenne)

# if image_plot:
#     plt.figure()
#     label_plot, plot =  displayImageLabel(Y_corr_moy,img_corrige_moy)

#     plt.show()


# # %% Classification supervisée

# # imgClassNum,roi = selectClassArea(img_corrige_moy)

# # plt.figure()
# # plt.imshow(imgClassNum)
# # plt.show()

# # np.save("num_class", imgClassNum)
# with open('num_class.npy', 'rb') as f:

#     mat_filtre = np.load(f)

# # Affichage de l'image corrigée


# # segmentation manuelle




# #%% Affichage des histogrammes et histo cumulé

# res = np.zeros(mat_filtre.shape)
# A=[]
# B=[]
# C=[]
# D=[]
# for i in range(img_corrige_moy.shape[0]):
#     for j in range(img_corrige_moy.shape[1]):
#             if mat_filtre[i,j] == 0:
#                 A.append(img_corrige_moy[i,j])
#             if mat_filtre[i,j] == 1:
#                 B.append(img_corrige_moy[i,j])
#             if mat_filtre[i,j] == 2:
#                 C.append(img_corrige_moy[i,j])
#             else:
#                 D.append(img_corrige_moy[i,j])
# A,B,C,D = np.array(A),np.array(B),np.array(C),np.array(D)
# if image_plot:
#     fig = plt.figure()
#     ax = fig.add_subplot(211)
#     ax.set_xlim(0,5)
#     sns.histplot(data=A, kde=True, bins=100,stat="density",color='blue')
#     sns.histplot(data=B, kde=True, bins=100,stat="density",color='green')
#     sns.histplot(data=C, kde=True, bins=100,stat="density",color='red')
#     sns.histplot(data=D, kde=True, bins=100,stat="density",color='purple')
#     # plt.hist(A,bins=100,density=True,label='Histogramme de la classe A',cumulative=True)
#     # plt.hist(B,bins=100,density=True,label='Histogramme de la classe B',cumulative=True)
#     # plt.hist(C,bins=100,density=True,label='Histogramme de la classe C',cumulative=True)
#     # plt.hist(D,bins=100,density=True,label='Histogramme de la classe D',cumulative=True)
#     ax= fig.add_subplot(212)
#     sns.histplot(data=A, kde=True, bins=100,stat="density",color='blue',cumulative=True)
#     sns.histplot(data=B, kde=True, bins=100,stat="density",color='green',cumulative=True)
#     sns.histplot(data=C, kde=True, bins=100,stat="density",color='red',cumulative=True)
#     sns.histplot(data=D, kde=True, bins=100,stat="density",color='purple',cumulative=True)

#     plt.legend()
#     plt.title("Histogrammes et densités de probabilitées")

# plt.show()

#%% Calcul des descripteurs stats et calcul des features
patchSize, patchShift = 128, 32
plt.figure()
for i in range(1,5):

    feat, patchParams = cptPatchesStats(img_corrige_moy,[patchSize*(i*2),patchSize*(i*2)],[patchShift*(i/4),patchShift*(i/4)])






    # classification par kmeans

        
    N = feat[:,0:3]# Size prend le nombre d'éléments dans la matrice (à ne pas confondre avec shape)
    # Nech = N[::50]
    # print(N.shape)
    ## Test sur un sous échantillon 
    # Application du K-Means : 
    model_kmeans = KMeans(n_clusters = 4, n_init=20)
    model_kmeans.fit(N)
    print(model_kmeans.labels_.shape)



    # calcul de la position des centres
    
    plt.subplot(2,2,i)
    plt.imshow(img_corrige_moy)
    y = np.mean(patchParams[:,0:2],axis=1)
    x = np.mean(patchParams[:,2:4],axis=1)
    # affichage
    plt.scatter(x,y,s=20,c=model_kmeans.labels_)
    plt.title("Classification non supervisée, patchSize ="+str(patchSize*(i*2))+" et patchShift ="+str(patchShift*(i*2)))
plt.show()
