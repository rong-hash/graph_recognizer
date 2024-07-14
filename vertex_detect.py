import cv2 as cv
import numpy as np
import pytesseract

def detect_vertices(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found or the file is not a valid image.")
    else:
        
        # Apply threshold to get a binary image
        _, binary = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Find all contours in the binary image
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # Prepare an image to draw contours
        contour_img = np.zeros_like(img)
        text_img = np.zeros_like(img)
        nodes = []
        # Recognize and fill the elliptical contours
        for i, contour in enumerate(contours):
            if len(contour) >= 50:  # Need at least 5 points to fit an ellipse
                ellipse = cv.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis_length = max(axes)
                minor_axis_length = min(axes)
                aspect_ratio = major_axis_length / minor_axis_length
                
                # Check if the contour is an ellipse by comparing aspect ratio or other properties
                if 1.0 <= aspect_ratio <= 1.1:  # Adjust this range according to your needs
                    
                    # draw the contour
                    cv.drawContours(contour_img, [contour], -1, (255, 255, 255), -1)

                    # find all the children of the contour
                    child = hierarchy[0][i][2]

                    coordinates = []
                    if child == -1:
                        continue
                    while child != -1:
                        # get the bounding box of the child contour
                        x, y, w, h = cv.boundingRect(contours[child])
                        # append the coordinates of the corner to coordinates list
                        coordinates.append((x, y))
                        coordinates.append((x + w, y + h))
                        # get the next child
                        child = hierarchy[0][child][0]

                    # find the left most x coordinate
                    left = min([x for x, y in coordinates])
                    # find the right most x coordinate
                    right = max([x for x, y in coordinates])
                    # find the top most y coordinate
                    top = min([y for x, y in coordinates])
                    # find the bottom most y coordinate
                    bottom = max([y for x, y in coordinates])

                    # use this bounding box as mask to get the text inside the ellipse
                    mask = np.zeros_like(img)
                    cv.rectangle(mask, (left, top), (right, bottom), (255, 255, 255), -1)

                    # Mask the image to only show the ellipse
                    masked_image = cv.bitwise_and(img, mask)

                    text_img[top:bottom, left:right] = masked_image[top:bottom, left:right]

                    # crop the image
                    masked_image = masked_image[top:bottom, left:right]

                    # find the background color of the masked image
                    background_color = np.median(masked_image)

                    # add a margin to the masked image with the background color
                    masked_image = cv.copyMakeBorder(masked_image, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=background_color)

                    # Use OCR to recognize the text inside the ellipse
                    text = pytesseract.image_to_string(masked_image, config='--psm 6')
                    nodes.append((i, center, (major_axis_length + minor_axis_length) / 4, text.strip()))

        # save the contour image
        cv.imwrite("output/vertex_image.png", contour_img)
        # save the text image
        cv.imwrite("output/vertex_label_image.png", text_img)

    return nodes