import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage import img_as_ubyte

from tqdm.notebook import tqdm

from utils import read_image, show_image

# BEGIN YOUR IMPORTS
IMAGES_PATH = os.path.join(".", "sudoku_puzzles", "train")
image_path = os.path.join(IMAGES_PATH, "image_0.jpg")
sudoku_image = read_image(image_path)
# END YOUR IMPORTS

def find_edges(image):
    """
    Args:
        image (np.array): (grayscale) image of shape [H, W]
    Returns:
        edges (np.array): binary mask of shape [H, W]
    """

    edges = cv2.Canny(image=image, threshold1=100, threshold2=200)
    
    return edges


def highlight_edges(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        highlited_edges (np.array): binary mask of shape [H, W]
    """

    kernel = np.ones((5,5), np.uint8)
    highlited_edges = cv2.dilate(edges, kernel, iterations=1)
    # dilate_image = cv2.dilate(edges, kernel, iterations=1)
    # highlited_edges = cv2.erode(dilate_image, kernel, iterations=1)

    return highlited_edges


def find_contours(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        contours (np.array, np.array, ...): tuple of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    """

    ret, thresh = cv2.threshold(edges, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours


def get_max_contour(contours):
    """
    Args:
        contours (np.array, np.array, ...): tuple of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    Returns:
        max_contour (np.array): an array of points (vertices) of the contour with the maximum area of shape [N, 1, 2]
    """

    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
    max_contour = c

    return max_contour


def order_corners(corners):
    """
    Args:
        corners (np.array): an array of corner points (corners) of shape [4, 2]
    Returns:
        ordered_corners (np.array): an array of corner points in order [top left, top right, bottom right, bottom left]
    """
    # BEGIN YOUR CODE

    corners = corners[corners[:, 1].argsort()]
    tc = corners[0:2]
    bc = corners[2:4]
    tc = tc[tc[:, 0].argsort()]
    bc = bc[bc[:, 0].argsort()]

    top_left = tc[0]
    top_right = tc[1]
    bottom_right = bc[1]
    bottom_left = bc[0]

    # END YOUR CODE

    ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left])

    return ordered_corners


def find_corners(contour, accuracy=0.101):
    """
    Args:
        contour (np.array): an array of points (vertices) of the contour of shape [N, 1, 2]
        accuracy (float): how accurate the contour approximation should be
    Returns:
        ordered_corners (np.array): an array of corner points (corners) of quadrilateral approximation of contour of shape [4, 2]
                                    in order [top left, top right, bottom right, bottom left]
    """

    # maybe this:  https://www.scaler.com/topics/contour-analysis-opencv/    
    accuracy = 0.01*cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, accuracy, True)
    corners = np.array([corners[0][0],
                        corners[1][0],
                        corners[2][0],
                        corners[3][0]])

    # to avoid errors
    if len(corners) != 4:
        corners = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

    ordered_corners = order_corners(corners)

    return ordered_corners


def rescale_image(image, scale=0.5):
    """
    Args:
        image (np.array): input image
        scale (float): scale factor
    Returns:
        rescaled_image (np.array): 8-bit (with range [0, 255]) rescaled image
    """
    # BEGIN YOUR CODE

    rescaled_image = rescale(image, scale, anti_aliasing=False)
    rescaled_image = img_as_ubyte(rescaled_image)
    
    # END YOUR CODE
    
    return rescaled_image


def gaussian_blur(image, sigma):
    """
    Args:
        image (np.array): input image
        sigma (float): standard deviation for Gaussian kernel
    Returns:
        blurred_image (np.array): 8-bit (with range [0, 255]) blurred image
    """
    # BEGIN YOUR CODE

    blurred_image = None # YOUR CODE
    
    # END YOUR CODE
    
    return blurred_image


def distance(point1, point2):
    """
    Args:
        point1 (np.array): n-dimensional vector
        point2 (np.array): n-dimensional vector
    Returns:
        distance (float): Euclidean distance between point1 and point2
    """
    # BEGIN YOUR CODE

    distance = np.linalg.norm(point1 - point2)
    
    # END YOUR CODE
    
    return distance


def warp_image(image, ordered_corners):
    """
    Args:
        image (np.array): input image
        ordered_corners (np.array): corners in order [top left, top right, bottom right, bottom left]
    Returns:
        warped_image (np.array): warped with a perspective transform image of shape [H, H]
    """
    # 4 source points
    top_left, top_right, bottom_right, bottom_left = ordered_corners

    # BEGIN YOUR CODE

    ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')     # the side length of the Sudoku grid based on distances between corners

    side = max([ distance(bottom_right, top_right), 
                 distance(top_left, bottom_left),
                 distance(bottom_right, bottom_left),   
                 distance(top_left, top_right) ])

    # what are the 4 target (destination) points?
    destination_points = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype='float32')

    # perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(ordered_corners, destination_points)

    # image warped using the found perspective transformation matrix
    warped_image = cv2.warpPerspective(image, transform_matrix, (int(side), int(side)))
    
    # END YOUR CODE

    assert warped_image.shape[0] == warped_image.shape[1], "height and width of the warped image must be equal"

    return warped_image


def frontalize_image(sudoku_image, pipeline):
    """
    Args:
        sudoku_image (np.array): input Sudoku image
        pipeline (Pipeline): Pipeline instance
    Returns:
        frontalized_image (np.array): warped with a perspective transform image of shape [H, H]
    """
    # BEGIN YOUR CODE

    image, ordered_corners = pipeline(sudoku_image, plot=True)
    
    frontalized_image = warp_image(image, ordered_corners)
    
    # END YOUR CODE

    return frontalized_image


def show_frontalized_images(image_paths, pipeline, figsize=(16, 12)):
    nrows = len(image_paths) // 4 + 1
    ncols = 4
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if len(axes.shape) == 1:
        axes = axes[np.newaxis, ...]

    for j in range(len(image_paths), nrows * ncols):
        axis = axes[j // ncols][j % ncols]
        show_image(np.ones((1, 1, 3)), axis=axis)
    
    for i, image_path in enumerate(tqdm(image_paths)):
        axis = axes[i // ncols][i % ncols]
        axis.set_title(os.path.split(image_path)[1])
        
        sudoku_image = read_image(image_path=image_path)
        frontalized_image = frontalize_image(sudoku_image, pipeline)

        show_image(frontalized_image, axis=axis, as_gray=True)
