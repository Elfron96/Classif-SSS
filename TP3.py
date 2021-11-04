#!/usr/bin/env python
# coding: utf-8

# # TP 3 : Prétraitements des données SSS
# #### Solana VIEL & Eva LASSAUGE - FISE 2021

# ## Introduction : 
# L'objectif du TP est de faire la classification des fonds marins à partir de données d'un sonar latéral
# Pour ce TP, vous exploiterez les données d'un sonar latéral Klein 5500 fonctionnant à 455 kHz.

# ## Partie 1 : Introduction 

# In[67]:


#pip install roipoly


# In[68]:


# Bibliothèques nécessaires : 

import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from roipoly import roipoly 
from displayImageLabel import displayImageLabel
from cptPatchesStats import cptPatchesStats
from scipy.io import loadmat
from selectClassArea import*

from displayImageLabel import*


# ### Question 1 : Chargement du fichier data_12 et affichage de l'image

# In[69]:


data = loadmat("12_data.mat")
data_img = data['img']


# plt.figure()
# plt.title("Affichage de la waterfall data_12")
# plt.imshow(data_img, aspect = "auto")
# plt.xlabel("Echantillon")
# plt.ylabel("N° de Pings")
# plt.colorbar(label ="Backscatter [dB]")
# plt.show()


# ### Question 2 : Description des différents constituants de l'image 
# 
# L'image que nous affichons correspond à la représentation temporelle des signaux accoutiques à l'issue d'un levé fait avec un sonar latéral. 
# 
# Sur cette image nous voyons sur l'intervalle de pings $[0;250]$ le trajet de l'onde dans l'eau, formant une zone dite "aveugle". Ensuite nous distinguons le fond marin. Celui-ci présente différentes morphologies, avec trois ensembles distincts. 
# Le premier, situé entre le ping 0 et 600 sur l'axe longitudinale (ordonnées) se caractérise par un fond homogène peut-être sableux. Le second situé entre le ping 600 et 800 est plus texturé, avec des rides de sables. Enfin le dernier situé entre le ping 800 et 1500, est lui aussi texturé mais les formes s'apparentent plus à de la vase modelée par les courants ou des roches. 

# ## Partie 2 : Classification non supervisée des fonds marins sans prétraitements 

# ### Question 1 : Segmentation de l'image avec la méthode des Kmeans 
# 
# Avant de procéder à la segmentation, nous allons mettre l'image sous forme d'un vecteur de N pixels * D descripteurs. Nous utiliserons l'intensité du pixel comme descripteur.
# 
# ### Question 2 : Analyse de la segmentation
# 
# Suite à la segmentation, nous voyons que les résultats ne sont pas satisfaisants, la classification détecte des classes en fonction du backscatter de manière verticale sur l'image. Nous comprenons ici que la classification pour le moment est dépendante de la portée ou de l'angle d'incidence. En effet, plus on s'éloigne, plus l'angle devient rasant et plus il y a de contraste au niveau du signal rétrodiffusé. 
# Nous voyons aussi que le K-Means détecte une classe pour les intensités faibles de la colonne d'eau. Nous pouvons dire que le résultat n'est pas satisfaisant, nous cherchons à classifier les fonds pas la colonne d'eau. 

# In[70]:


N = data_img.reshape((data_img.size, 1))# Size prend le nombre d'éléments dans la matrice (à ne pas confondre avec shape)
N_ech = N[::50]

## Test sur un sous échantillon 
# Application du K-Means : 
model_kmeans = KMeans(n_clusters = 3 )
model_kmeans.fit(N_ech)

#Prédictions des classes : 
Y_ech = model_kmeans.predict(N_ech)

# Affichage des résultats : 
# plt.figure()
# plt.scatter(N_ech, Y_ech, c= Y_ech, label = "Classes K-Means")
# plt.title("Affichage des classes déterminées par K-Means")
# plt.xlabel("Pixels")
# plt.ylabel("Classes")
# plt.show()

## K-Means sur toute l'image : 
# On garde l'a priori avec 3 classes : 
model_kmeans.fit(N)
Y = model_kmeans.predict(N)

label, plot =  displayImageLabel(Y, data_img)


# ### Question 3 : Tronquage de la colonne d'eau 
# 
# Sans la colonne d'eau, nous voyons que la classification change un peu. La présence de la colonne engendrait lors de classification la détection de faibles valeurs d'intensité. Ici nous voyons que ce n'est plus le cas. Nous pouvons dire que la colonne d'eau ne perturbe plus la classification. Le contraste de l'image entre les classes semble un peu plus élevé. 

# In[71]:


img_wc = data_img[:,245:]
N_wc = img_wc.reshape((img_wc.size, 1))


# plt.figure()
# plt.title("Affichage de la waterfall data_12 sans colonne d'eau")
# plt.imshow(img_wc, aspect = "auto")
# plt.xlabel("Echantillon")
# plt.ylabel("N° de Pings")
# plt.colorbar(label ="Backscatter [dB]")
# plt.show()


# # Application de K-Means : 

model_kmeans.fit(N_wc)
Y_wc = model_kmeans.predict(N_wc)

Clasification_wc, plot =  displayImageLabel(Y_wc, img_wc)


# ## Partie 3 : Classification non-supervisée des fonds avec prétraitements
# 
# Dans cette partie, nous allons coder la méthode de correction d'amplitude afin de prétaiter les données. Pour ce faire nous allons moyenner les amplitudes de tous les pings pour chaque échantillon. 

# ### Question 1 : Calcul de  la  moyenne  en  fonction  du  numéro  d’échantillon  sur  l’ensemble  des  pings.  Affichage du résultat en fonction  du numéro d'échantillon. Correction de l’image en divisant par la moyenne calculée précédemmment

# In[72]:


## Correction de l'image par moyennage simple et affichage

## on reprend l'image de départ (data_img):
moy = np.mean(img_wc, axis = 0) # moyenne selon axis = 0 c'est-à-dire selon l'ensemble des pings (colonne 1)

## Division de l'image par la moyenne des pings 

img_cor = img_wc / moy


# ### Question 2 : Représentation de  l’image  originale  (avec  colorbar)  et  la  courbe  de  variation  en  fonction  du  numéro d’échantillon ainsi que les versions corrigées (image avec colorbar et courbe).
# 
# Nous voyons que quand nous n'appliquons pas de correction d'amplitude sur l'image, le contraste est décroissant avec l'augmentation de la distance. Cette tendance se confirme par l'allure de la courbe de variation de l'intensité en fonction du numéro d'échantillon, en effet l'amplitude diminue avec la distance. 
# 
# Une fois la correction d'amplitude appliquée, le contraste diminue, il n'est plus fonction de la distance. Le but du prétraitement est de déterminer une courbe qui redresse les niveaux entre le début et la fin de la portée. 
# 
# Quand on applique le K-Means sur l'image prétraitée, les classes ne sont plus observées selon la distance mais bien selon les différents fonds. Cependant, utiliser l'intensité comme seul descripteur ne suffit pas à bien détecter les différents fonds. 

# In[73]:


## Affichage de l'image originale : 
# plt.figure()
# plt.title("Affichage de la waterfall data_12")
# plt.imshow(img_wc, aspect = "auto")
# plt.xlabel(" N°d'échantillon")
# plt.ylabel("N° de Pings")
# plt.colorbar(label ="Backscatter [dB]")
# plt.show()

# ## Affichage des résultats : 
# plt.figure()
# plt.plot(moy)
# plt.xlabel("Numéro d'échantillon")
# plt.ylabel("Moyenne sur l'ensemble des pings")
# plt.title("Evolution de la moyenne sur l'ensemble des pings en fonction du numéro d'échantillon")
# plt.show()

## Affichage de l'image corrigée : 

# plt.figure()
# plt.imshow(img_cor)
# plt.title("Image corrigée de la moyenne")
# plt.xlabel("N° d'échantillon")
# plt.ylabel("Moyenne sur l'ensemble des pings")
# plt.colorbar(label =  "Backscatter [dB]")
# plt.show()


# # In[74]:


# ### Courbe de l'évolution de la moyenne sur l'image corrigée 
# plt.figure()
# plt.plot(np.mean(img_cor, axis = 0))
# plt.xlabel("Numéro d'échantillon")
# plt.ylabel("Moyenne sur l'ensemble des pings")
# plt.title("Evolution de la moyenne sur l'ensemble des pings en fonction du numéro d'échantillon sur l'image corrigé")
# plt.show()


# In[75]:


## K-Mean sur l'image corrigée : 

N_cor = img_cor.reshape((img_cor.size, 1))
# # Application de K-Mean : 

model_kmeans.fit(N_cor)
Y_cor = model_kmeans.predict(N_cor)

Clasification_wc, plot =  displayImageLabel(Y_cor, img_cor)


# ## Partie 4 : Classification supervisée des fonds marins avec prétraitements 

# ### Question 1 : Sélection des zones présentant des fonds différents dans l'image 

# In[ ]:





# In[77]:


num_class, roi = selectClassArea(img_cor)

np.save("num_class", num_class)

# In[ ]:




