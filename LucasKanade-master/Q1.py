"""
Danish Waheed
CAP5415 - Fall 2017

This Python program computes Lucas-Kanade optical flow without a Gaussian Pyramid
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
from LK import compute_flow_map, lucas_kanade, hornShunckFlow
np.seterr(divide='ignore', invalid='ignore')

# Setting the size of the window which will contain the flow map for both image sets
plt.figure(figsize=(15, 10))

list_figures=[#'motion01.512.tiff',
              'motion02.512.tiff',
              'motion03.512.tiff','motion04.512.tiff',
              'motion05.512.tiff',
              #'motion06.512.tiff',
              #'motion07.512.tiff','motion08.512.tiff',
              ]
#list_figures=['basket1.png','basket2.png', 'basket3.png','basket4.png']
#list_figures=['foto1.jpg','foto2.jpg', 'foto3.jpg', 'foto4.jpg', 'foto5.jpg', 'foto6.jpg', 'foto7.jpg']

count=1
for i in range(len(list_figures)-1): 
    # Reading in the first set of images
    image1 = cv2.cvtColor(cv2.imread('./gif_basquete/'+list_figures[i]), cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(cv2.imread('./gif_basquete/'+list_figures[i+1]), cv2.COLOR_RGB2GRAY)
    print('#------------------------------#-----------------------------#')
    u1, v1 = lucas_kanade(image1, image2, 5)
    u2, v2 = hornShunckFlow(image1, image2, 1)

    flow_map1 = compute_flow_map(u1,v1, 8)
    flow_map2 = compute_flow_map(v1, v2, 8)

    plt.imsave("LK_flow_map_motion_win_3_"+str(count)+'.png', flow_map1, cmap='gray')
    plt.imshow(flow_map1, cmap='gray')
    #plt.show()


    image_mask1 = image1 + flow_map1
    plt.imsave("LK_with_mask_motion_win_3_"+str(count)+'.png', image_mask1, cmap='gray')
    plt.imshow(image_mask1, cmap='gray')
    plt.show()
    plt.clf()
    
    plt.imsave("HS_flow_map_motion_alpha_4_"+str(count)+'.png', flow_map2, cmap='gray')
    plt.imshow(flow_map2, cmap='gray')
    #plt.show()


    image_mask2 = image1 + flow_map2
    plt.imsave("HS_with_mask_motion_alpha_4_"+str(count)+'.png', image_mask2, cmap='gray')
    plt.imshow(image_mask1, cmap='gray')
    plt.show()
    plt.clf()
    count+=1


