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
    
def dft_2D(pixels):
    return np.fft.fft2(pixels)
    
# Aplicacao da tranformada DCT 2D 
def dct_2D(pixels):
    return fftpack.dct(fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')
    
# Aplicacao da tranformada DCT 2D 
def dst_2D(pixels):
    return fftpack.dst(fftpack.dst(pixels.T, norm='ortho').T, norm='ortho')
    
#%%
    
pastas = [#['Escala de Cinza','./benchmarks/gray8bit/'],
          ['Diversas','./benchmarks/misc/']]
          
          
for database, path in pastas:
    resultados = []
    list_images = load_images(path)
    for j in list_images:
        image = get_image(list_images, j)
        img_size1 = image.shape[0]
        img_size2 = image.shape[1] 
        
        print(j,'\n')
        
#        plt.xticks(())
#        plt.yticks(())
#        plt.title("Original",fontsize=18)
#        plt.imshow(get_reconstructed_image(image), cmap=plt.cm.gray)
#        plt.show()
#        plt.clf()
#       
        ruido = np.random.normal(loc=128,scale=20,size=(img_size1,img_size2))
        ruido = ruido-np.min(ruido)
        print(ruido)
        image_noise = image+ruido #random_noise(image, mode='speckle') 
        # DCT
        image_dct = dct_2D(image)
        image_dct_abs = abs(image_dct)
        image_dct_noise = dct_2D(image_noise)
        image_dct_noise_abs = abs(image_dct_noise) 
        image_dct_log = np.log10(image_dct_abs)
        image_dct_noise_log = np.log10(image_dct_noise_abs)
        
        max_image_dct_abs = np.max(image_dct_abs)
        max_image_dct_noise_abs = np.max(image_dct_noise_abs)
        max_image_dct_log = np.max(image_dct_log)
        max_image_dct_noise_log = np.max(image_dct_noise_log)
        
        min_image_dct_abs = np.min(image_dct_abs)
        min_image_dct_noise_abs = np.min(image_dct_noise_abs)
        min_image_dct_log = np.min(image_dct_log)
        min_image_dct_noise_log = np.min(image_dct_noise_log)
        
        mean_image_dct_abs = np.mean(image_dct_abs)
        mean_image_dct_noise_abs = np.mean(image_dct_noise_abs)
        mean_image_dct_log = np.mean(image_dct_log)
        mean_image_dct_noise_log = np.mean(image_dct_noise_log)
        
#        plt.title("Módulo DCT - Sem ruído")        
#        plt.imshow(get_reconstructed_image(image_dct_abs), cmap='jet')
#        plt.colorbar()
#        plt.show()
#        plt.clf()
        
#        plt.title("Módulo DCT - Com ruído")        
#        plt.imshow(get_reconstructed_image(image_dct_noise_abs), cmap='jet')
#        plt.colorbar()
#        plt.show()
#        plt.clf()
        
#        plt.title("Log Módulo DCT - Sem ruído")        
#        plt.imshow(get_reconstructed_image(image_dct_log), cmap='jet')
#        plt.colorbar()
#        plt.show()
#        plt.clf()
        
#        plt.title("Log Módulo DCT - Com ruído")        
#        plt.imshow(get_reconstructed_image(image_dct_noise_log), cmap='jet')
#        plt.colorbar()
#        plt.show()
#        plt.clf()
          
          
        plt.figure(figsize=(14, 8))
        plt.subplot(2, 3, 1)
        plt.xticks(())
        plt.yticks(())
        plt.title("Original",fontsize=16)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.subplot(2, 3, 2)
        plt.title("Módulo DCT - Sem ruído", fontsize=16)        
        plt.imshow(get_reconstructed_image(image_dct_abs), cmap='jet')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.subplot(2, 3, 3)
        plt.title("Log Módulo DCT - Sem ruído", fontsize=16)        
        plt.imshow(get_reconstructed_image(image_dct_log), cmap='jet')
        #plt.colorbar()
        
        plt.subplot(2, 3, 4)
        plt.xticks(())
        plt.yticks(())
        plt.title("Com Ruído",fontsize=16)
        plt.imshow(image_noise, cmap=plt.cm.gray)
        
        plt.subplot(2, 3, 5)
        plt.title("Módulo DCT - Com ruído", fontsize=16)        
        plt.imshow(get_reconstructed_image(image_dct_noise_abs), cmap='jet')
        #plt.colorbar()
        
        plt.subplot(2, 3, 6)
        plt.title("Log Módulo DCT - Com ruído", fontsize=16)        
        plt.imshow(get_reconstructed_image(image_dct_noise_log), cmap='jet')
        #plt.colorbar()
        plt.tight_layout()
        
        plt.show()
        plt.clf()
        
        

#        n = j.split('/')[-1]
#        n = n.replace('.png','')
#        plt.savefig("./Resultados/Questao2/"+str(database)+"_"+str(n)+"_"+"bilinear"+".png")
#        plt.clf()
        
        # DST
        image_dst = dst_2D(image)
        image_dst_abs = abs(image_dst)
        image_dst_noise = dst_2D(image_noise)
        image_dst_noise_abs = abs(image_dst_noise) 
        image_dst_log = np.log10(image_dst_abs)
        image_dst_noise_log = np.log10(image_dst_noise_abs)
        
        max_image_dst_abs = np.max(image_dst_abs)
        max_image_dst_noise_abs = np.max(image_dst_noise_abs)
        max_image_dst_log = np.max(image_dst_log)
        max_image_dst_noise_log = np.max(image_dst_noise_log)
        
        min_image_dst_abs = np.min(image_dst_abs)
        min_image_dst_noise_abs = np.min(image_dst_noise_abs)
        min_image_dst_log = np.min(image_dst_log)
        min_image_dst_noise_log = np.min(image_dst_noise_log)
        
        mean_image_dst_abs = np.mean(image_dst_abs)
        mean_image_dst_noise_abs = np.mean(image_dst_noise_abs)
        mean_image_dst_log = np.mean(image_dst_log)
        mean_image_dst_noise_log = np.mean(image_dst_noise_log)
        
        plt.title("Módulo DST - Sem ruído")        
        plt.imshow(get_reconstructed_image(image_dst_abs), cmap='jet')
        plt.colorbar()
        plt.show()
        plt.clf()
        
        plt.title("Módulo DST - Com ruído")        
        plt.imshow(get_reconstructed_image(image_dst_noise_abs), cmap='jet')
        plt.colorbar()
        plt.show()
        plt.clf()
        
        plt.title("Log Módulo DST - Sem ruído")        
        plt.imshow(get_reconstructed_image(image_dst_log), cmap='jet')
        plt.colorbar()
        plt.show()
        plt.clf()
        
        plt.title("Log Módulo DST - Com ruído")        
        plt.imshow(get_reconstructed_image(image_dst_noise_log), cmap='jet')
        plt.colorbar()
        plt.show()
        plt.clf()
        
        # DFT
        image_dft = dft_2D(image)
        image_dft_abs = abs(image_dft)
        image_dft_noise = dft_2D(image_noise)
        image_dft_noise_abs = abs(image_dft_noise) 
        image_dft_log = np.log10(image_dft_abs)
        image_dft_noise_log = np.log10(image_dft_noise_abs)
        
        
        max_image_dft_abs = np.max(image_dft_abs)
        max_image_dft_noise_abs = np.max(image_dft_noise_abs)
        max_image_dft_log = np.max(image_dft_log)
        max_image_dft_noise_log = np.max(image_dft_noise_log)
        
        min_image_dft_abs = np.min(image_dft_abs)
        min_image_dft_noise_abs = np.min(image_dft_noise_abs)
        min_image_dft_log = np.min(image_dft_log)
        min_image_dft_noise_log = np.min(image_dft_noise_log)
        
        mean_image_dft_abs = np.mean(image_dft_abs)
        mean_image_dft_noise_abs = np.mean(image_dft_noise_abs)
        mean_image_dft_log = np.mean(image_dft_log)
        mean_image_dft_noise_log = np.mean(image_dft_noise_log)
        
        plt.title("Módulo DFT - Sem ruído")        
        plt.imshow(get_reconstructed_image(image_dft_abs), cmap='jet')
        plt.colorbar()
        plt.show()
        plt.clf()
        
        plt.title("Módulo DFT - Com ruído")        
        plt.imshow(get_reconstructed_image(image_dft_noise_abs), cmap='jet')
        plt.colorbar()
        plt.show()
        plt.clf()
        
        plt.title("Log Módulo DFT - Sem ruído")        
        plt.imshow(get_reconstructed_image(image_dft_log), cmap='jet')
        plt.colorbar()
        plt.show()
        plt.clf()
        
        plt.title("Log Módulo DFT - Com ruído")        
        plt.imshow(get_reconstructed_image(image_dft_noise_log), cmap='jet')
        plt.colorbar()
        plt.show()
        plt.clf()

        

    
    