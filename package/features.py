import cv2
import numpy as np



def auto_crop_image(image_path):
    img = cv2.imread(image_path)
    a = int(img.shape[0] * 3 / 100)
    b = int(img.shape[0] * 3 / 100)
    c = int(img.shape[1] * 2 / 100)
    img = img[a:img.shape[0] - b, c:img.shape[1]-c]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10,10,7)
    ret, gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    gray = cv2.dilate(gray, np.ones((5                    , 5), np.uint8), iterations=5)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(gray))     # cropping image to text region only
    img = img[y:y+h, x:x+w]
    return img

def get_letter_slant(image_path):
    # Load the image
    img = auto_crop_image(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges = cv2.dilate(edges, (5, 5), iterations=5)

    # Apply the Hough transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=50)

    # Draw the detected lines and calculate their inclination
    line_angles = []
    try:
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
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            inclination = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            line_angles.append(inclination)
    except:
        return None, img

    return np.mean(line_angles)

def get_line_slant(image_path):
    # Load the image
    img = auto_crop_image(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 3)
    # Apply binary threshold
    ret, thresh = cv2.threshold(gray, 130, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, (5, 5), iterations=7)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find lines using Hough transform
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # Calculate the angles of the lines
    angles = []
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            angles.append(angle)
    except:
        return None, img

    # Calculate the average angle of the lines
    avg_angle = np.mean(angles)

    # Rotate the image by the average angle
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), avg_angle, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Draw the lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return avg_angle, img