# -*- coding: utf-8 -*-
"""
Processamento grafico - trabalho 1
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

def load_images(path):
    list_of_images = glob.glob(path+'*.png')
    return list_of_images


# Abertura da imagem da URL passada como parametro
def get_image(list_images, k):
    #file_descriptor = urllib2.urlopen(image_url)
    #image_file = io.BytesIO(file_descriptor.read())
    image = Image.open(k)
    #img_color = image.resize(size, 1)
    img_grey = image.convert('L')
    img = np.array(img_grey, dtype=np.float)
    return img


# Reconstrucao da imagem
def get_reconstructed_image(raw):
    img = raw.clip(0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    return img


#==============================================================================    

#     
pastas = [#['Escala de Cinza','./benchmarks/gray8bit/'],
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
            


