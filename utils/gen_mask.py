'''
python3 utils/gen_mask.py
'''
import cv2
import numpy as np

#init
H = 502
W = 1024
STEP = 120
SIZE = 5

# define some block of code

if __name__ == "__main__":

    print('Gen Mask')

    blank = np.zeros((H,W),np.uint8) + 255
    
    # draw horizontal lines
    for i in range(0,W,STEP):
        blank[:,i:i+SIZE] = 0
    
    # draw vertical lines
    for j in range(0,H,STEP):
        blank[j:j+SIZE,:] = 0

    cv2.imshow('res',blank)
    cv2.waitKey(0)
    cv2.destroyAllWindows()