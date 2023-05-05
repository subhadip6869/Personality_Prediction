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
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
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

    return np.median(line_angles)

def get_line_slant(image_path):
    # Load the image
    img = auto_crop_image(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 3)
    # Apply binary threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    thresh = cv2.dilate(thresh, (5, 5), iterations=10)

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
    avg_angle = np.median(angles)

    # Rotate the image by the average angle
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), avg_angle, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Draw the lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return avg_angle, img


def get_letter_size(image_path):
    img = auto_crop_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # thresh = cv2.threshold(gray, 130, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, (5, 5), iterations=5)
    thresh = cv2.dilate(thresh, (5, 5), iterations=5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        total_area.append(area)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    avg_area = np.median(np.array(total_area))
    return avg_area, img


def gap_between_words(image_path):
    # Load the image
    img = auto_crop_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply morphological operations to remove noise and connect broken components
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.dilate(morph, (5, 5), iterations=7)

    # Find contours of all connected components
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Calculate word spacing
    word_spacing = []
    for i in range(len(contours)-1):
        x, y, w, h = cv2.boundingRect(contours[i])
        next_x = cv2.boundingRect(contours[i+1])[0]
        spacing = next_x - (x + w)
        word_spacing.append(abs(spacing))

    # Draw word spacing on the original image
    for i in range(len(contours)-1):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(img, (x + w, y), (x + w + word_spacing[i], y + h), (0, 255, 0), 2)

    return np.median(np.array(word_spacing)), img
