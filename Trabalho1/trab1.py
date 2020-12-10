# -*- coding: utf-8 -*-
"""
Processamento grafico - trabalho 1
Alunas: Gisele Goulart e Marvelúcia Almeida
Programa de Pós GRaduação em Modelagem Computacional
"""
# Pacotes utilizados
import io
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack
import math
import IPython
from sklearn.metrics import mean_squared_error
from skimage import data
from skimage.color import rgb2gray
from numpy.linalg import svd
from skimage import img_as_ubyte,img_as_float
import glob
import seaborn as sns
import os
from skimage.transform import rescale

#%%

def load_images(path):
    list_of_images = glob.glob(path+'*.png')
    return list_of_images


# Abertura da imagem da URL passada como parametro
def get_image(list_images, k):
    image = Image.open(k)
    img_grey = image.convert('L')
    img = np.array(img_grey, dtype=np.float)
    return img


# Reconstrucao da imagem
def get_reconstructed_image(raw):
    img = raw.clip(0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    return img

#%%
#==============================================================================    
# Questao 1
# Quantizacao
#     
pastas = [['Escala de Cinza','./benchmarks/gray8bit/'],
          ['Diversas','./benchmarks/misc/']]


# Loop no vetor de imagens

for database, path in pastas:
    resultados = []
    list_images = load_images(path)
    for j in list_images:
        for intensidade in [63,31,15,7,1]:
            pixels = get_image(list_images, j)
            img_size1 = pixels.shape[0]
            img_size2 = pixels.shape[1]
            
            print(j,"--", img_size1,"x", img_size2, "\n")
            #print(pixels, "\n")
            novas_intensidades = pixels.copy()
            for linha in range(img_size1):
                for coluna in range(img_size2):
                    #if(pixels[linha,coluna]>63):
                        novas_intensidades[linha,coluna] = round((intensidade/255)*pixels[linha,coluna])
                    
            #print(novas_intensidades, "\n") 
            n = j.split('/')[-1]
            n = n.replace('.png','')
            reconstructed_image = get_reconstructed_image(novas_intensidades)
            plt.imshow(reconstructed_image, cmap=plt.cm.gray)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            #plt.show()
            plt.savefig(str(database)+"_"+str(n)+"_"+str(intensidade)+".png")
            plt.clf()
            

#%%

# Questao 2
# Interpolacao

pastas = [['Escala de Cinza','./benchmarks/gray8bit/'],
          ['Diversas','./benchmarks/misc/']]

for database, path in pastas:
    resultados = []
    list_images = load_images(path)
    for j in list_images:
        image = get_image(list_images, j)
        img_size1 = image.shape[0]
        img_size2 = image.shape[1]
            
        print(j,"--", img_size1,"x", img_size2, "\n")
       
        image_rescaled = rescale(image, 0.5,order=1)
        reconstructed_image1 = get_reconstructed_image(image_rescaled)
#        
        image_rescaled2 = rescale(image_rescaled, 2,order=1)
        reconstructed_image2 = get_reconstructed_image(image_rescaled2)

        
        
        plt.figure(figsize=(14, 12))
        plt.subplot(1, 3, 1)
        plt.xticks(())
        plt.yticks(())
        plt.title("Original",fontsize=18)
        plt.imshow(image, cmap=plt.cm.gray)
        
        plt.subplot(1, 3, 2)
        plt.xticks(())
        plt.yticks(())
        plt.title("Redução",fontsize=18)
        plt.imshow(reconstructed_image1, cmap=plt.cm.gray)
        
        plt.subplot(1, 3, 3)
        plt.xticks(())
        plt.yticks(())
        plt.title("Ampliação",fontsize=18)
        plt.imshow(reconstructed_image2, cmap=plt.cm.gray)
        plt.tight_layout()
        #plt.show()
        n = j.split('/')[-1]
        n = n.replace('.png','')
        plt.savefig("./Resultados/Questao2/"+str(database)+"_"+str(n)+"_"+"bilinear"+".png")
        plt.clf()
        
#%%
        
#  Questao 3
#  Subtracao de imagens
        
pastas = [['Sequencial','./benchmarks/sequencial/'],
          ]
          
for database, path in pastas:
    resultados = []
    list_images = load_images(path)
    print(list_images)
    image1 = get_image(list_images, list_images[3])
    image2 = get_image(list_images, list_images[2])
    image3 = get_image(list_images, list_images[0])
    image4 = get_image(list_images, list_images[1])
    img_size1 = image1.shape[0]
    img_size2 = image1.shape[1]
            
    
    f2_f1 = image1.copy()
    f3_f2 = image1.copy()
    f4_f3 = image1.copy()    

    for i in range(img_size1):
        for j in range(img_size2):
            f2_f1[i,j] = image2[i,j]-image1[i,j]
            f3_f2[i,j] = image3[i,j]-image2[i,j]
            f4_f3[i,j] = image4[i,j]-image3[i,j]

    min2_1 = f2_f1.min()         
    min3_2 = f3_f2.min()
    min4_3 = f4_f3.min()
    
    for i in range(img_size1):
        for j in range(img_size2):
            f2_f1[i,j] = f2_f1[i,j] - min2_1
            f3_f2[i,j] = f3_f2[i,j] - min3_2
            f4_f3[i,j] = f4_f3[i,j] - min4_3

    max2_1 = f2_f1.max()         
    max3_2 = f3_f2.max()
    max4_3 = f4_f3.max()
    
    for i in range(img_size1):
        for j in range(img_size2):
            f2_f1[i,j] = 255*(f2_f1[i,j]/max2_1)
            f3_f2[i,j] = 255*(f3_f2[i,j]/max3_2)
            f4_f3[i,j] = 255*(f4_f3[i,j]/max4_3)
    
    
    reconstructed_image = get_reconstructed_image(f2_f1)
    plt.imshow(reconstructed_image, cmap=plt.cm.gray)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.clf()
    
    reconstructed_image = get_reconstructed_image(f3_f2)
    plt.imshow(reconstructed_image, cmap=plt.cm.gray)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.clf()
    
    reconstructed_image = get_reconstructed_image(f4_f3)
    plt.imshow(reconstructed_image, cmap=plt.cm.gray)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.clf()





        
        
        
        
        
        