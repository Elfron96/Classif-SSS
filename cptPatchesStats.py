#%% import useful packages or functions
import numpy as np
import scipy.stats as stats
from getPatchesParams import getPatchesParams

#%% def cptPatchesStats
def cptPatchesStats(img, patchSize, patchShift):
    ''' Fonction permettant de calculer des stats sur l'ensemble de patches 
    extrait de l'image img.
        Un patch est défini par sa taille patchSize (nbLig,nbCol) et son 
        déplacement caractérisé par patchShift (nbLig,nbCol). 
        Le résultat est rendu dans une matrice feat de dimension 
        nbPatches x nbFeat
    '''

    #%% Locales
    # cte 
    
    # var
    NB_FEAT = 16; # stats
    
    #%% calcul des indices de début et de fin de lignes et de colonnes de patchs
    sizeOfImg = img.shape
    patchesParams = getPatchesParams(sizeOfImg, patchSize, patchShift)
    
    nbWdw = patchesParams.shape[0]
    
    #%% calcul des features
    feat = np.zeros((nbWdw,NB_FEAT))
    
    for iWdw in np.arange(0,nbWdw):
    
        # parametres courants
        curIndLig = np.arange(patchesParams[iWdw,0],patchesParams[iWdw,1])
        curIndCol = np.arange(patchesParams[iWdw,2],patchesParams[iWdw,3])
        patch = img[curIndLig,curIndCol]
    
        # stats
        feat[iWdw,:] = cptStats(patch[:])

    return feat, patchesParams


#%% def cptStats

''' Calcul des statistiques descriptives de donnees

% INPUT :
%   computeStats(sig) calcule les statistiques descriptives (moyenne, etc.)
%   sur les données sig. 
%
% OUTPUT :
%   outVar = computeStats(...) renvoie le résultat dans outVar structure de
%   données englobant les champs:
%       - mean, var, std, med, min, max, range, q1, q3, q25, q75, q97, q99, 
%         skew, kurt 
%
% REMARKS : 
%
% EXAMPLES :
%
% SEE ALSO   : 
% AUTHORS    : gilles.lechenadec@alyotech.fr
%--------------------------------------------------------------------------
''' 
def cptStats(sig, valNaN=None, flagVector=False):

    # --------------------------------------------------------------------------
    # HISTORIQUE DEVELOPPEMENT
    #   01/04/09 - GLC - creation
    # --------------------------------------------------------------------------
    
    
    #%% Traitements
    q1,q3,q25,q50,q75,q97,q99 = np.percentile(sig, [1,3,25,50,75,97,99])
    
    feat = np.array([[np.mean(sig),
                      np.var(sig),
                      np.std(sig),
                      q50,
                      np.min(sig),
                      np.max(sig),
                      np.ptp(sig),
                      stats.skew(sig),
                      stats.kurtosis(sig),
                      q1,
                      q3,
                      q25,
                      q75,
                      q97,
                      q99,
                      q75-q25]])
        

    
    return feat