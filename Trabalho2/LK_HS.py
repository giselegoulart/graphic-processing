# -*- coding: utf-8 -*-    
"""-
Trabalho de Processamento Gráfico 2020 - 2020/2
Alunas: Gisele Goulart e Marvelúcia Almeida

Adaptações de
 https://github.com/dmarkatia/LucasKanade/blob/master/LK.py 
 e
https://stackoverflow.com/questions/27904217/horn-schunck-optical-flow-implementation-issue

Implementação dos métodos Horn Shunck e Lucas Kanade para o cálculo do fluxo ótico
"""


import numpy as np
import cv2
import matplotlib.pylab as plt

np.seterr(divide='ignore', invalid='ignore')

def compute_flow_map(u, v, gran=8):
    flow_map = np.zeros(u.shape)

    for y in range(flow_map.shape[0]):
        for x in range(flow_map.shape[1]):

            if y % gran == 0 and x % gran == 0:
                dx = 10 * int(u[y, x])
                dy = 10 * int(v[y, x])

                if dx > 0 or dy > 0:
                    cv2.arrowedLine(flow_map, (x, y), (x + dx, y + dy), 255, 1)

    return flow_map

def takePixel(img, i, j):
    i = i if i >= 0 else 0
    j = j if j >= 0 else 0    
    i = i if i < img.shape[0] else img.shape[0] - 1
    j = j if j < img.shape[1] else img.shape[1] - 1
    return img[i, j]
    
# Derivadas numericas do paper original: http://people.csail.mit.edu/bkph/papers/Optical_Flow_OPT.pdf
def xDer(img1, img2):
    res = np.zeros_like(img1)
    for i in list(range(res.shape[0])):
        for j in list(range(res.shape[1])):
            sm = 0
            sm += int(takePixel(img1, i,     j + 1)) - int(takePixel(img1, i,     j))
            sm += int(takePixel(img1, i + 1, j + 1)) - int(takePixel(img1, i + 1, j))
            sm += int(takePixel(img2, i,     j + 1)) - int(takePixel(img2, i,     j))
            sm += int(takePixel(img2, i + 1, j + 1)) - int(takePixel(img2, i + 1, j))
            sm /= 4.0
            res[i, j] = sm
    return res

def yDer(img1, img2):
    res = np.zeros_like(img1)
    for i in list(range(res.shape[0])):
        for j in list(range(res.shape[1])):
            sm = 0
            sm += int(takePixel(img1, i + 1, j    )) - int(takePixel(img1, i, j    ))
            sm += int(takePixel(img1, i + 1, j + 1)) - int(takePixel(img1, i, j + 1))
            sm += int(takePixel(img2, i + 1, j    )) - int(takePixel(img2, i, j    ))
            sm += int(takePixel(img2, i + 1, j + 1)) - int(takePixel(img2, i, j + 1))
            sm /= 4.0
            res[i, j] = sm
    return res

def tDer(img, img2):
    res = np.zeros_like(img)
    for i in list(range(res.shape[0])):
        for j in list(range(res.shape[1])):
            sm = 0
            for ii in list(range(i, i + 2)):
                for jj in list(range(j, j + 2)):
                    sm += int(takePixel(img2, ii, jj)) - int(takePixel(img, ii, jj))
            sm /= 4.0
            res[i, j] = sm
    return res

# filtro de media
def average(img):
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(img.astype(np.float32), -1, kernel)
    
# Implementação Lucas Kanade
def lucas_kanade(im1, im2, win):
    Ix = np.zeros(im1.shape)
    Iy = np.zeros(im1.shape)
    It = np.zeros(im1.shape)

    Ix[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    It[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]

    params = np.zeros(im1.shape + (5,))
    params[..., 0] = cv2.GaussianBlur(Ix * Ix, (5, 5), 3)
    params[..., 1] = cv2.GaussianBlur(Iy * Iy, (5, 5), 3)
    params[..., 2] = cv2.GaussianBlur(Ix * Iy, (5, 5), 3)
    params[..., 3] = cv2.GaussianBlur(Ix * It, (5, 5), 3)
    params[..., 4] = cv2.GaussianBlur(Iy * It, (5, 5), 3)

    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])

    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)
    Ixx = win_params[..., 0]
    Iyy = win_params[..., 1]
    Ixy = win_params[..., 2]
    Ixt = -win_params[..., 3]
    Iyt = -win_params[..., 4]
    
    
    M_det = Ixx * Iyy - Ixy ** 2
    temp_u = Iyy * (-Ixt) + (-Ixy) * (-Iyt)
    temp_v = (-Ixy) * (-Ixt) + Ixx * (-Iyt)
    op_flow_x = np.where(M_det != 0, temp_u / M_det, 0)
    op_flow_y = np.where(M_det != 0, temp_v / M_det, 0)

    u[win + 1: -1 - win, win + 1: -1 - win] = op_flow_x[:-1, :-1]
    v[win + 1: -1 - win, win + 1: -1 - win] = op_flow_y[:-1, :-1]
    
    # Cálculo da derivada material
    der_mat = It+Ix*u+Iy*v
    
    norma_der = np.linalg.norm(der_mat,'fro')
    print('Norma da Derivada Material LK: ', norma_der)
    plt.clf()
    plt.imshow(der_mat, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()
    plt.clf()
    return u, v
    
#Implementação Método Horn Shunck
def hornShunckFlow(img1, img2, alpha):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    Idx = xDer(img1, img2)
    Idy = yDer(img1, img2)
    Idt = tDer(img1, img2)

    u = np.zeros_like(img1)
    v = np.zeros_like(img1)

    #100 iterations 
    for iteration in list(range(100)):
        u0 = np.copy(u)
        v0 = np.copy(v)

        uAvg = average(u0)
        vAvg = average(v0)
        # '*', '+', '/' operations in numpy works component-wise
        u = uAvg - 1.0/(alpha**2 + Idx**2 + Idy**2) * Idx * (Idx * uAvg + Idy * vAvg + Idt)
        v = vAvg - 1.0/(alpha**2 + Idx**2 + Idy**2) * Idy * (Idx * uAvg + Idy * vAvg + Idt)


    # Cálculo da derivada material        
    der_mat = Idt+Idx*u+Idy*v
    
    norma_der = np.linalg.norm(der_mat,'fro')
    print('Norma da Derivada Material HS: ', norma_der)  
    plt.clf()
    plt.imshow(der_mat, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.show()
    plt.clf()
    
    return u, v
    
    