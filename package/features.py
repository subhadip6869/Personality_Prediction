import cv2 as cv
import numpy as np

def get_letter_slant(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img = cv.medianBlur(img, 5)
    img = cv.threshold(img, 120, 255, cv.THRESH_BINARY_INV)[1]
    img = cv.dilate(img, (5, 5), iterations=7)
    # Apply a morphological closing operation to fill any gaps between letters
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=kernel)

    # Find the contours of the connected components in the binary image
    contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    # Fit a line to each contour and calculate the angle of the line
    angles = []
    for contour in contours:
        # Fit a line to the contour using the least squares method.
        rows,cols = img.shape[:2]
        [vx,vy,x,y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
        # Calculate the angle of the line with respect to the horizontal
        angle = np.rad2deg(np.arctan2(vy, vx))
        angles.append(angle)

    # Take the average of the angles to estimate the slant of the text
    slant = np.mean(angles)
    return slant



def get_line_slant(image_path):
    # Load the image
    img = cv.imread(image_path)

    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # Apply the Hough transform
    lines = cv.HoughLines(edges, rho=1, theta=np.pi/180, threshold=50)

    # Draw the detected lines and calculate their inclination
    line_angles = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        inclination = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        line_angles.append(inclination)

    return np.mean(line_angles)