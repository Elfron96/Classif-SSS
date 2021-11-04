#%% load useful packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tkinter.messagebox
import matplotlib
from roipoly import RoiPoly

#matplotlib.interactive(True)

#plt.ion()

#%% load useful packages
def selectClassArea( img):
    '''
    # img est l'image des données
    '''

    ## Affichage des données
    hFig = plt.figure()
    plt.cla()
    ax = plt.imshow(img)

    #    axis tight
    q = np.percentile(img[:], [5, 95])
    plt.colorbar()
    plt.xlabel('Numero d\'échantillon')
    plt.ylabel('Numero de ping')
    ax.set_clim(q[0], q[1])
    ax.set_cmap('copper')
    plt.title('image normalisée sans colonne d\'eau')
    plt.pause(.1)

    #
    # R = tkinter.messagebox.askyesno('Question à l''utilisateur', 'Voulez-vous créer d''autres classes ?');
    R = input('Voulez-vous créer des classes (yes/no) ? ')
    col = 'bgrcmyk'

    # nbClass = 2;
    nbClass = 1
    cont = R
    imgClassNum = np.zeros(img.shape)
    roi = []
    while R.lower()=='yes':

        # select arera
        my_roi = RoiPoly(color=col[nbClass])  # draw new ROI in red color
        # my_roi = roipoly(fig=hFig,roicolor=col[nbClass])  # draw new ROI in red color


        # mask total
        imgClassNum = imgClassNum + nbClass*my_roi.getMask(img)

        # update
        roi.append(my_roi)
        my_roi = []

        # update
        nbClass = nbClass + 1

        # Continue?
        # R = tkinter.messagebox.askyesno('Question à l''utilisateur', 'Voulez-vous créer d''autres classes ?');
        R = input('Voulez-vous créer des classes (yes/no) ? ')

        if R.lower()=='yes':
            plt.figure(num=hFig.number)
            plt.cla()
            ax = plt.imshow(img)

            #    axis tight
            q = np.percentile(img[:], [5, 95])
            plt.colorbar()
            plt.xlabel('Numero d\'échantillon')
            plt.ylabel('Numero de ping')
            ax.set_clim(q[0], q[1])
            ax.set_cmap('copper')
            plt.title('image normalisée sans colonne d\'eau')
            plt.pause(.1)

            # display area
            for r in roi:
                r.display_roi()
            # my_roi.display_roi()

    # Fin
    plt.close(hFig);


#    plt.ioff()
   
    return imgClassNum,roi


    # Masque des zones d'intéret
    # plt.figure(2);
    # plt.cla()
    # for r in [my_roi1, my_roi2, my_roi3]
    #     r.display_roi()
