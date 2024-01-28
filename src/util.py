import math
import cv2
import numpy as np
from typing import List, NamedTuple
from PIL import Image, ImageDraw

class Keypoint(NamedTuple):
    x: float
    y: float
    visible: bool

def translate_to_keyppoints(pose, leave_out_invisible=False):
    if leave_out_invisible:
        return [Keypoint(x, y, True) for x, y in pose]
    return [Keypoint(x, y, True) if 0.0 < x < 1.0 and 0.0 < y < 1.0 else None for x, y in pose]


def draw_bodypose(canvas: np.ndarray, keypoints: List[Keypoint], translation=(0, 0), show_invisible=False) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, _ = canvas.shape


    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    tx, ty = translation

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if not show_invisible and (keypoint1 is None or keypoint2 is None):
            continue

        Y = np.array([keypoint1.x + tx, keypoint2.x + tx]) * float(W)
        X = np.array([keypoint1.y + ty, keypoint2.y + ty]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if not show_invisible and keypoint is None:
            continue

        x, y = keypoint.x + tx, keypoint.y + ty
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas

def create_image_grid_with_borders(images, rows, cols, border=1, border_color='white'):
    if not images:
        raise ValueError("No images to compose.")

    # Assuming all images are the same size
    w, h = images[0].size

    # Size of the grid, accounting for borders
    grid_width = cols * w + (cols - 1) * border
    grid_height = rows * h + (rows - 1) * border

    # Create a new image with a white background
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_img)

    for i, img in enumerate(images):
        # Calculate grid position
        grid_x = (i % cols) * (w + border)
        grid_y = (i // cols) * (h + border)

        # Paste the image into position
        grid_img.paste(img, (grid_x, grid_y))

        # Draw vertical borders
        if (i % cols) != (cols - 1):  # Avoid drawing on the last column
            draw.line([(grid_x + w, grid_y), (grid_x + w, grid_y + h)], fill=border_color, width=border)

        # Draw horizontal borders
        if (i // cols) != (rows - 1):  # Avoid drawing on the last row
            draw.line([(grid_x, grid_y + h), (grid_x + w, grid_y + h)], fill=border_color, width=border)

    return grid_img