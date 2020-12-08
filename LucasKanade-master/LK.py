"""
Danish Waheed
CAP5415 - Fall 2017

This Python program is the supporting library for Q1 and Q2
"""

import numpy as np
import cv2
from scipy import signal
import sys

# This sets the program to ignore a divide error which does not affect the outcome of the program
np.seterr(divide='ignore', invalid='ignore')

'''
TODO: Add comments for this method
'''
def reduce(image, level=1):
    result = np.copy(image)

    for _ in range(level - 1):
        result = cv2.pyrDown(result)

    return result


'''
TODO: Add comments for this method
'''
def expand(image, level=1):
    return cv2.pyrUp(np.copy(image))


'''
TODO: Add comments for this method
'''
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


'''
TODO: Add comments for this method
'''
def lucas_kanade(im1, im2, win=7):
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
    print('Tam u', u.shape,u,'\n',u[9], u[10],'\n')
    print('Tam v', v.shape,v,'\n',v[9], v[10],'\n')
    der_mat = It+Ix*u+Iy*v
    print('Derivada Material LK:', der_mat)
    
    return u, v
    
def takePixel(img, i, j):
    i = i if i >= 0 else 0
    j = j if j >= 0 else 0    
    i = i if i < img.shape[0] else img.shape[0] - 1
    j = j if j < img.shape[1] else img.shape[1] - 1
    return img[i, j]
    
#Numerical derivatives from original paper: http://people.csail.mit.edu/bkph/papers/Optical_Flow_OPT.pdf
def xDer(img1, img2):
    res = np.zeros_like(img1)
    for i in list(range(res.shape[0])):
        for j in list(range(res.shape[1])):
            sm = 0
            sm += takePixel(img1, i,     j + 1) - takePixel(img1, i,     j)
            sm += takePixel(img1, i + 1, j + 1) - takePixel(img1, i + 1, j)
            sm += takePixel(img2, i,     j + 1) - takePixel(img2, i,     j)
            sm += takePixel(img2, i + 1, j + 1) - takePixel(img2, i + 1, j)
            sm /= 4.0
            res[i, j] = sm
    return res

def yDer(img1, img2):
    res = np.zeros_like(img1)
    for i in list(range(res.shape[0])):
        for j in list(range(res.shape[1])):
            sm = 0
            sm += takePixel(img1, i + 1, j    ) - takePixel(img1, i, j    )
            sm += takePixel(img1, i + 1, j + 1) - takePixel(img1, i, j + 1)
            sm += takePixel(img2, i + 1, j    ) - takePixel(img2, i, j    )
            sm += takePixel(img2, i + 1, j + 1) - takePixel(img2, i, j + 1)
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
                    sm += takePixel(img2, ii, jj) - takePixel(img, ii, jj)
            sm /= 4.0
            res[i, j] = sm
    return res
    
def average(img):
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(img.astype(np.float32), -1, kernel)
    
def hornShunckFlow(img1, img2, alpha):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    Idx = xDer(img1, img2)
    Idy = yDer(img1, img2)
    Idt = tDer(img1, img2)

    u = np.zeros_like(img1)
    v = np.zeros_like(img1)

    #100 iterations enough for small example
    for iteration in list(range(100)):
        u0 = np.copy(u)
        v0 = np.copy(v)

        uAvg = average(u0)
        vAvg = average(v0)
        # '*', '+', '/' operations in numpy works component-wise
        u = uAvg - 1.0/(alpha**2 + Idx**2 + Idy**2) * Idx * (Idx * uAvg + Idy * vAvg + Idt)
        v = vAvg - 1.0/(alpha**2 + Idx**2 + Idy**2) * Idy * (Idx * uAvg + Idy * vAvg + Idt)
        if  iteration % 10 == 0:
            print('iteration', iteration, np.linalg.norm(u - u0) + np.linalg.norm(v - v0))
            
    der_mat = Idt+Idx*u+Idy*v
    print('Derivada Material HS:', der_mat)        
    return u, v
    
    
    #Funciona, mas os vetores ficam bastante embolados
def optical_flow(I1g, I2g, window_size, tau=1e-2):
 
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, int(I1g.shape[0]-w)):
        for j in range(w, int(I1g.shape[1]-w)):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            b = np.reshape(It, (It.shape[0],1)) # get b here
            A = np.vstack((Ix, Iy)).T # get A here
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu =  nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
            u[i,j]=nu[0]
            v[i,j]=nu[1]
 
    return u,v
    
    # Nao estao funcionando
def optical_flow2(img1, img2, window_size, tau=1e-2):
    # calculate gradients along x direction
    filter_x = np.transpose(np.array([[-1., -1.], [1., 1.]]))
    f_x1 = signal.convolve2d(img1, filter_x, mode = 'same')
    f_x2 = signal.convolve2d(img2, filter_x, mode = 'same')    
    f_x = f_x1 + f_x2
    
    # calculate gradients along y direction
    filter_y = np.array([[-1., -1.], [1., 1.]])
    f_y1 = signal.convolve2d(img1, filter_y, mode = 'same')
    f_y2 = signal.convolve2d(img2, filter_y, mode = 'same')
    f_y = f_y1 + f_y2
    
    # calculate gradient along t direction
    filter_t1 = np.array([[-1., -1.], [-1., -1.]])
    f_t1 = signal.convolve2d(img1, filter_t1, mode = 'same')
    filter_t2 = np.array([[1., 1.], [1., 1.]])
    f_t2 = signal.convolve2d(img2, filter_t2, mode = 'same')
    f_t = f_t1 + f_t2
    
    filter_lap = np.array([[0, -1./4, 0], [-1./4, 1., -1./4], [0, -1./4, 0]])
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    
    
#    # solve overdetermined system by assuming that optical flow is constant in a 3 by 3 window centered on each pixel
#    for i in list(range(1, u.shape[0])):
#        for j in list(range(1, u.shape[1])):
#            f_x9 = f_x[i - 1:i + 2, j - 1:j + 2].flatten()
#            f_y9 = f_y[i - 1:i + 2, j - 1:j + 2].flatten()
#            ft = f_t[i - 1:i + 2, j - 1:j + 2].flatten()
#            A = np.vstack((f_x9, f_y9)).T
#            res = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T, A)), A.T), ft)
#            u[i, j] = res[0]
#            v[i, j] = res[1]
    
    
    # try least squares fit instead, we expect it is less error, smoother field
    for i in list(range(1, u.shape[0])):
        for j in list(range(1, u.shape[1])):
            u_num = - np.sum(np.power(f_y[i - 1:i + 2, j - 1:j + 2], 2)) * \
            np.sum(np.multiply(f_x[i - 1:i + 2, j - 1:j + 2], f_t[i - 1:i + 2, j - 1:j + 2])) + \
            np.sum(np.multiply(f_x[i - 1:i + 2, j - 1:j + 2], f_y[i - 1:i + 2, j - 1:j + 2])) * \
            np.sum(np.multiply(f_y[i - 1:i + 2, j - 1:j + 2], f_t[i - 1:i + 2, j - 1:j + 2]))
            u_denom = np.sum(np.power(f_x[i - 1:i + 2, j - 1:j + 2], 2)) * \
            np.sum(np.power(f_y[i - 1:i + 2, j - 1:j + 2], 2)) - \
            np.power(np.sum(np.multiply(f_x[i - 1:i + 2, j - 1:j + 2], f_y[i - 1:i + 2, j - 1:j + 2])), 2)
            
            v_num = np.sum(np.multiply(f_x[i - 1:i + 2, j - 1:j + 2], f_t[i - 1:i + 2, j - 1:j + 2])) * \
            np.sum(np.multiply(f_x[i - 1:i + 2, j - 1:j + 2], f_y[i - 1:i + 2, j - 1:j + 2])) - \
            np.sum(np.power(f_x[i - 1:i + 2, j - 1:j + 2], 2)) * \
            np.sum(np.multiply(f_y[i - 1:i + 2, j - 1:j + 2], f_t[i - 1:i + 2, j - 1:j + 2]))
            v_denom = np.sum(np.power(f_x[i - 1:i + 2, j - 1:j + 2], 2)) * \
            np.sum(np.power(f_y[i - 1:i + 2, j - 1:j + 2], 2)) - \
            np.power(np.sum(np.multiply(f_x[i - 1:i + 2, j - 1:j + 2], f_y[i - 1:i + 2, j - 1:j + 2])), 2)
            
            u[i, j] = u_num / u_denom
            v[i, j] = v_num / v_denom
        return u,v