"""
OpenCV Color Picker and Basic Color Detector

This script lets the user click on an image to pick a color.
It creates an HSV mask around the selected pixel, shows the masked result,
and estimates a basic color name from the average HSV values in the masked area.

Controls:
- Left click on the image: pick a color
- Click the RESET button: clear selection and return to normal
- Press 'q': quit the program
"""

import cv2
import numpy as np


def stackImages(scale, imgArray):
    """
    Resize and stack multiple images into a single output image.

    This function supports:
    - a 1D list of images: [img1, img2, img3]
    - a 2D list of images: [[img1, img2], [img3, img4]]

    It also:
    - resizes images to match the first image
    - converts grayscale images to BGR
    - applies the given scale before stacking

    Parameters:
        scale (float): Scale factor for resizing images
        imgArray (list): List or nested list of OpenCV images

    Returns:
        numpy.ndarray: Final stacked image
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)

    if rowsAvailable:
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]

        for x in range(0, rows):
            for y in range(0, cols):

                # Resize image if size is different from the first image
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y],
                        (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                        None,
                        scale,
                        scale
                    )

                # Convert grayscale image to BGR so stacking works correctly
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows

        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])

        ver = np.vstack(hor)

    else:
        for x in range(0, rows):

            # Resize image if size is different from the first image
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x],
                    (imgArray[0].shape[1], imgArray[0].shape[0]),
                    None,
                    scale,
                    scale
                )

            # Convert grayscale image to BGR so stacking works correctly
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

        hor = np.hstack(imgArray)
        ver = hor

    return ver


# =========================
# Image setup
# =========================
path = "color_picker/sources/image.png"
img = cv2.imread(path)
img = cv2.resize(img, (700, 700))
imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# =========================
# Global state
# =========================
lower = None
upper = None
mode = "normal"
button_pos = (10, 10, 120, 50)


def get_color(h, s, v):
    """
    Return a basic color name based on HSV values.

    Parameters:
        h (float): Hue
        s (float): Saturation
        v (float): Value

    Returns:
        str: Approximated color name
    """
    if v < 50:
        return "black"
    elif s < 40 and v > 200:
        return "white"
    elif s < 40:
        return "gray"
    elif h < 10 or h >= 170:
        return "red"
    elif h < 20:
        return "orange"
    elif h < 35:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 125:
        return "blue"
    elif h < 150:
        return "purple"
    elif h < 170:
        return "pink"
    else:
        return "unknown"


def pick_color(event, x, y, flags, param):
    """
    Mouse callback function.

    Left-click behavior:
    - If the click is inside the RESET button, clear the current selection.
    - Otherwise, pick the HSV color at the clicked pixel and create a range
      around it for masking.

    Parameters:
        event: OpenCV mouse event
        x (int): x-coordinate of mouse click
        y (int): y-coordinate of mouse click
        flags: OpenCV flags
        param: Extra parameters
    """
    global lower, upper, mode

    bx, by, bw, bh = button_pos

    if event == cv2.EVENT_LBUTTONDOWN:

        # RESET button
        if bx <= x <= bx + bw and by <= y <= by + bh:
            mode = "normal"
            lower = None
            upper = None
            print("Back to normal")
            return

        # Pick color from clicked pixel
        h, s, v = map(int, imgHsv[y, x])
        lower = np.array([max(0, h - 10), max(0, s - 40), max(0, v - 40)])
        upper = np.array([min(179, h + 10), min(255, s + 40), min(255, v + 40)])
        mode = "mask"
        print("Picked HSV:", h, s, v)


cv2.namedWindow("Image")
cv2.setMouseCallback("Image", pick_color)
output_open = False

while True:
    display = img.copy()

    # Draw RESET button
    bx, by, bw, bh = button_pos
    cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 0, 255), -1)
    cv2.putText(
        display,
        "RESET",
        (bx + 10, by + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if lower is not None and upper is not None:
        mask = cv2.inRange(imgHsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        hsvDisplay = imgHsv[mask > 0]
        avg_h = np.mean(hsvDisplay[:, 0])
        avg_s = np.mean(hsvDisplay[:, 1])
        avg_v = np.mean(hsvDisplay[:, 2])
        color_name = get_color(avg_h, avg_s, avg_v)

        cv2.putText(
            result,
            f"Color: {color_name}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        stack = stackImages(0.7, [mask, result])
        cv2.imshow("Output", stack)
        output_open = True

    else:
        if output_open:
            cv2.destroyWindow("Output")
            output_open = False

    cv2.imshow("Image", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()