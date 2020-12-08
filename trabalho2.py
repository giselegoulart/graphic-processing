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
from skimage.util import random_noise
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
    
pastas = [['Escala de Cinza','./benchmarks/gray8bit/'],
          ['Diversas','./benchmarks/misc/']]
          
          
for database, path in pastas:
    resultados = []
    list_images = load_images(path)
    for j in list_images:
        image = get_image(list_images, j)
        img_size1 = image.shape[0]
        img_size2 = image.shape[1] 

        image_noise = random_noise(image,seed=10)     
        
        plt.xticks(())
        plt.yticks(())
        plt.title("Original",fontsize=18)
        plt.imshow(get_reconstructed_image(image), cmap=plt.cm.gray)
        plt.show()
        plt.clf()
    
    