import os

from tqdm import tqdm

from skimage.io import imsave

from pipeline import get_test_pipeline

from utils import read_image
from recognition import TEMPLATES_PATH
from frontalization import frontalize_image
from recognition import get_sudoku_cells

# BEGIN YOUR IMPORTS


# END YOUR IMPORTS

IMAGES_PATH = os.path.join(".", "sudoku_puzzles", "train")

# BEGIN YOUR CODE

"""
create dict of cell coordinates like in this example

CELL_COORDINATES = {"image_0.jpg": {'1': (0, 0),
                                    '2': (1, 1)},
                    "image_2.jpg": {'1': (2, 3),
                                    '3': (2, 1),
                                    '9': (5, 6)}}
"""

CELL_COORDINATES = {
    "image_0.jpg": {
        '8': (0,5),
        '3': (1,3),
        '4': (1,4),
        '8': (1,8),
        '9': (2,0),
        '3': (2,1),
        '7': (2,4),
        '4': (2,7),
        '2': (2,8),
        '3': (3,0),
        '4': (3,3),
        '7': (3,8),
        '5': (4,0),
        '7': (4,1),
        '6': (4,4),
        '4': (5,1),
        '7': (5,3),
        '8': (5,4),
        '6': (5,6),
        '9': (5,7),
        '5': (5,8),
        '4': (6,0),
        '1': (6,4),
        '5': (6,7),
        '9': (6,8),
        '5': (7,3),
        '1': (7,7),
        '6': (7,8),
        '5': (8,2),
        '9': (8,5),
        '4': (8,6),
        '8': (8,7),
        '3': (8,8),
    },
    "image_1.jpg": {
        '2': (0,1),
        '8': (1,0),
        '3': (1,1),
        '1': (1,3),
        '5': (1,4),
        '7': (1,6),
        '2': (1,7),
        '6': (1,8),
        '4': (2,0),
        '3': (2,4),
        '1': (2,8),
        '5': (3,0),
        '8': (3,3),
        '2': (3,4),
        '4': (3,7),
        '7': (4,1),
        '9': (4,3),
        '2': (4,6),
        '1': (4,7),
        '7': (5,3),
        '8': (5,7),
        '9': (5,8),
        '1': (6,1),
        '9': (6,4),
        '8': (6,5),
        '7': (7,0),
        '3': (7,3),
        '6': (7,7),
        '9': (8,0),
        '7': (8,4),
        '3': (8,8),
    },
    # "image_2.jpg": {
    #     '': (0,0),
    #     '': (0,1),
    #     '': (0,2),
    #     '': (0,3),
    #     '': (0,4),
    #     '': (0,5),
    #     '': (0,6),
    #     '': (0,7),
    #     '': (0,8),
    #     '': (1,0),
    #     '': (1,1),
    #     '': (1,2),
    #     '': (1,3),
    #     '': (1,4),
    #     '': (1,5),
    #     '': (1,6),
    #     '': (1,7),
    #     '': (1,8),
    #     '': (2,0),
    #     '': (2,1),
    #     '': (2,2),
    #     '': (2,3),
    #     '': (2,4),
    #     '': (2,5),
    #     '': (2,6),
    #     '': (2,7),
    #     '': (2,8),
    #     '': (3,0),
    #     '': (3,1),
    #     '': (3,2),
    #     '': (3,3),
    #     '': (3,4),
    #     '': (3,5),
    #     '': (3,6),
    #     '': (3,7),
    #     '': (3,8),
    #     '': (4,0),
    #     '': (4,1),
    #     '': (4,2),
    #     '': (4,3),
    #     '': (4,4),
    #     '': (4,5),
    #     '': (4,6),
    #     '': (4,7),
    #     '': (4,8),
    #     '': (5,0),
    #     '': (5,1),
    #     '': (5,2),
    #     '': (5,3),
    #     '': (5,4),
    #     '': (5,5),
    #     '': (5,6),
    #     '': (5,7),
    #     '': (5,8),
    #     '': (6,0),
    #     '': (6,1),
    #     '': (6,2),
    #     '': (6,3),
    #     '': (6,4),
    #     '': (6,5),
    #     '': (6,6),
    #     '': (6,7),
    #     '': (6,8),
    #     '': (7,0),
    #     '': (7,1),
    #     '': (7,2),
    #     '': (7,3),
    #     '': (7,4),
    #     '': (7,5),
    #     '': (7,6),
    #     '': (7,7),
    #     '': (7,8),
    #     '': (8,0),
    #     '': (8,1),
    #     '': (8,2),
    #     '': (8,3),
    #     '': (8,4),
    #     '': (8,5),
    #     '': (8,6),
    #     '': (8,7),
    #     '': (8,8),
    # }
}

# END YOUR CODE

def main():
    os.makedirs(TEMPLATES_PATH, exist_ok=True)
    
    pipeline = get_test_pipeline()

    for file_name, coordinates_dict in CELL_COORDINATES.items():
        image_path = os.path.join(IMAGES_PATH, file_name)
        sudoku_image = read_image(image_path=image_path)
    
        # BEGIN YOUR CODE

        frontalized_image = frontalize_image(sudoku_image, pipeline=pipeline) # YOUR CODE
        sudoku_cells = get_sudoku_cells(frontalized_image) # YOUR CODE
        
        # END YOUR CODE

        for digit, coordinates in tqdm(coordinates_dict.items(), desc=file_name):
            digit_templates_path = os.path.join(TEMPLATES_PATH, digit)
            os.makedirs(digit_templates_path, exist_ok=True)
            
            digit_template_path = os.path.join(digit_templates_path, f"{os.path.splitext(file_name)[0]}_{digit}.jpg")
            imsave(digit_template_path, sudoku_cells[*coordinates])


if __name__ == "__main__":
    main()
