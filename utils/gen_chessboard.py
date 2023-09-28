'''
python3 utils/gen_chessboard.py
'''
import cv2
import numpy as np

def gen_mask():
    # Set the size of the checkerboard and the size of each square
    board_size = (1024, 502)  # Width and height of the checkerboard
    square_size = 100  # Width and height of each square

    # Create a white background
    background_color = (255)  # White color in BGR format
    image = np.full((board_size[1], board_size[0]), background_color, dtype=np.uint8)

    # Create the checkerboard pattern
    for i in range(0, board_size[0], square_size * 2):
        for j in range(0, board_size[1], square_size * 2):
            cv2.rectangle(image, (i, j), (i + square_size, j + square_size), (0), -1)

    for i in range(square_size, board_size[0], square_size * 2):
        for j in range(square_size, board_size[1], square_size * 2):
            cv2.rectangle(image, (i, j), (i + square_size, j + square_size), (0), -1)
            
    return image

image = gen_mask()

# Display the generated checkerboard pattern
cv2.imshow('Checkerboard', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the checkerboard pattern to a file
# cv2.imwrite('checkerboard.png', image)