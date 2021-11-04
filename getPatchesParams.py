#%% import useful packages
import numpy as np


#%% def getPatchesParams
def getPatchesParams(sizeOfImg, patchSize, patchShift):
    ''' Fonction permettant de calculer les indices (lignes et colonnes) de début
        et de fin quand on veut découper une image en patchs  de taille patchSize 
        (nbLig,nbCol) avec des déplacements caractérisés par patchShift 
        (nbLig,nbCol). Le résultat est dans patchesParams une matrice:
         - avec une ligne pour chaque petite image
         - avec 4 colonnes: numéros de début et de fin de ligne et numéros de
         début et de fin de la colonne.
    '''
    
    #%% Locales
    
    # var
    nbLin = sizeOfImg[0]
    nbCol = sizeOfImg[1]
    
    # nombre de pas à effectuer
    wdwNbLin   = int(np.fix((nbLin - patchSize[0])/patchShift[0]) + 1)
    wdwNbCol   = int(np.fix((nbCol - patchSize[1])/patchShift[1]) + 1)
    
    #%% Traitements
    patchesParams = np.zeros((wdwNbLin*wdwNbCol,4), dtype=int) # indices de début et de fin de lignes et de début et de fin de colonnes 
    
    compt = 0
    for j in np.arange(0, nbLin-patchSize[0]+1, patchShift[0]):
        
        for k in np.arange(0, nbCol-patchSize[1]+1, patchShift[1]):

            patchesParams[compt,0] = j
            patchesParams[compt,1] = j+patchSize[0]-1
            patchesParams[compt,2] = k
            patchesParams[compt,3] = k+patchSize[1]-1
    
            compt = compt + 1
            
            
    return patchesParams
        
        
    
    
    
    
