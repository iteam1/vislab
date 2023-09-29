'''
https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
python3 utils/add_noisy.py assets/vase.jpg
'''
import os
import cv2
import sys
import random
import numpy as np

# init
img_path = sys.argv[1]
    
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
    
if __name__ == "__main__":
    
    print("Add noisy to image")
    
    img = cv2.imread(img_path)
    
    img_noise = sp_noise(img,0.05)
    
    cv2.imshow("noisy", img_noise)
    cv2.waitKey(0)
    cv2.destroyAllWindows()