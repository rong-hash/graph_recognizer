import cv2 as cv
import numpy as np
import pytesseract
from utils import get_closest_node

def detect_edge(image_path, node_list):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found or the file is not a valid image.")
    else:
        
        # Apply threshold to get a binary image
        _, binary = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        masked_image = binary.copy()
        # Find all contours in the binary image
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # Prepare an image to draw contours
        contour_img = np.zeros_like(img)
        nodes = []
        # Recognize and fill the elliptical contours
        for i, contour in enumerate(contours):
            if len(contour) >= 50:  # Need at least 5 points to fit an ellipse
                # draw the contour
                temp = np.zeros_like(img)
                ellipse = cv.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis_length = max(axes)
                minor_axis_length = min(axes)
                aspect_ratio = major_axis_length / minor_axis_length
                
                # Check if the contour is an ellipse by comparing aspect ratio or other properties
                if 1.0 <= aspect_ratio <= 1.1:  # Adjust this range according to your needs
                    # mask the ellipse in the original image
                    mask = np.zeros_like(img)
                    cv.ellipse(mask, ellipse, (255, 255, 255), -1)
                    mask_not = cv.bitwise_not(mask)
                    masked_image = cv.bitwise_and(masked_image, mask_not)
                    kernel_size = 5  # This determines the diameter of the circle
                    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    dilated = cv.dilate(mask, kernel, iterations=1)
                    end_loop = 0
                    while cv.bitwise_and(masked_image, dilated).any() and (not end_loop):
                        overlap = cv.bitwise_and(masked_image, dilated)
                        # judge if the overlap area is a circle
                        contours, _ = cv.findContours(overlap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if len(contour) >= 50:
                                ellipse = cv.fitEllipse(contour)
                                (center, axes, orientation) = ellipse
                                major_axis_length = max(axes)
                                minor_axis_length = min(axes)
                                aspect_ratio = major_axis_length / minor_axis_length
                                if 1.0 <= aspect_ratio <= 1.1:
                                    dilated_not = cv.bitwise_not(dilated)
                                    masked_image = cv.bitwise_and(masked_image, dilated_not)
                                    dilated = cv.dilate(dilated, np.ones((5, 5), np.uint8), iterations=1)
                                else:
                                    # do one more dilation to erase the noise
                                    dilated_not = cv.bitwise_not(dilated)
                                    masked_image = cv.bitwise_and(masked_image, dilated_not)
                                    end_loop = 1
                                    break
                            else:
                                # do one more dilation to erase the noise
                                dilated_not = cv.bitwise_not(dilated)
                                masked_image = cv.bitwise_and(masked_image, dilated_not)
                                end_loop = 1
                                break


        arrow_image, text_image = get_filter_arrow_image(masked_image, node_list)
        
        if text_image is not None:
            cv.imwrite("output/edge_label_image.png", text_image)
        edge_list = []
        if arrow_image is not None:
            arrow_info, arrow_info_image = get_arrow_info(arrow_image, text_image)
            cv.imwrite("output/edge_info_image.png", arrow_info_image)

            # iterate through the arrow info to get the edge tip and tail
            for _, tip, tail, text in arrow_info:
                # find the closest node to the tip and tail
                tip_node = get_closest_node(tip, node_list)
                tail_node = get_closest_node(tail, node_list)
                if tip_node is not None and tail_node is not None:
                    edge_list.append((tip_node[0], tail_node[0], text[0][0]))
                
        return edge_list
    
def get_filter_arrow_image(threshold_image, node_list):
    blank_image = np.zeros_like(threshold_image)
    text_blank_image = np.zeros_like(threshold_image)
    contours, _ = cv.findContours(threshold_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    text_contour_list = []
    arrow_contour_list = []
    for contour in contours:
        is_arrow = 0
        # find the distance between the contour and node list
        for node in node_list:
            distance = abs(cv.pointPolygonTest(contour, node[1], True))
            if abs(distance - node[2]) < 0.1 * node[2]:
                cv.drawContours(blank_image, [contour], -1, 255, -1)
                is_arrow = 1
                break
        if is_arrow == 0:
            # draw the original shape that the contour represents
            x, y, w, h = cv.boundingRect(contour)
            roi = threshold_image[y:y+h, x:x+w]
            # place this shape on the blank image
            text_blank_image[y:y+h, x:x+w] = roi
    
    
    return blank_image, text_blank_image

def get_arrow_info(arrow_image, text_image):
    contours, hierarchy = cv.findContours(arrow_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    arrow_info = []
    arrow_info_image = cv.cvtColor(arrow_image.copy(), cv.COLOR_GRAY2BGR)
    if hierarchy is not None:

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)

            # Crop the image around the contour
            croppedImg = arrow_image[y:y+h, x:x+w]

            # Extend the borders for skeletonization
            borderSize = 5
            croppedImg = cv.copyMakeBorder(croppedImg, borderSize, borderSize, borderSize, borderSize, cv.BORDER_CONSTANT)


            # Compute the skeleton
            skeleton = cv.ximgproc.thinning(croppedImg, None, 1)
            skeleton = skeleton / 255

            # Apply convolution to identify endpoints
            endpoint_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
            img_filtered = cv.filter2D(skeleton, -1, endpoint_kernel)


            # Find endpoints
            endpoints = np.where(img_filtered == 11)
            points = np.column_stack((endpoints[1], endpoints[0]))  # format to x, y

            # Apply K-means clustering to the endpoints
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv.kmeans(np.float32(points), 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)



            # Determine which cluster is the tip (which has more points)
            tip_label = 0 if np.count_nonzero(labels == 0) > np.count_nonzero(labels == 1) else 1

            # Extract coordinates of the tip and tail
            tip = tuple(centers[tip_label].astype(int))
            tail = tuple(centers[1 - tip_label].astype(int))

            # Correct the coordinates based on the crop and border offset
            tip = (tip[0] + x - borderSize, tip[1] + y - borderSize)
            tail = (tail[0] + x - borderSize, tail[1] + y - borderSize)

            # Draw the tip and tail on the original image
            cv.circle(arrow_info_image, tip, radius = 5, color = (0, 0, 255), thickness = -1)
            cv.circle(arrow_info_image, tail, radius = 5, color = (0, 255, 0), thickness = -1)
            arrow_info.append([contour, tip, tail, None])
    
    # get the contour of text imaage
    contours, hierarchy = cv.findContours(text_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    more_than_one_text_arrow = []
    for contour in contours:
        if len(contour) < 10:
            continue
        x, y, w, h = cv.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        margin_width = int(1 / 4 * w)
        text_img = text_image[y-margin_width:y+h+margin_width, x-margin_width:x+w+margin_width] if y-margin_width >= 0 and x-margin_width >= 0 else text_image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(text_img, config='--psm 6')
        # find the closest arrow to the text
        min_distance = float("inf")
        closest_arrow = None
        for arrow in arrow_info:
            arrow_contour = arrow[0]
            distance = abs(cv.pointPolygonTest(arrow_contour, center, True))
            if distance < min_distance:
                min_distance = distance
                closest_arrow = arrow
        if closest_arrow is not None:
            if closest_arrow[3] is None:
                closest_arrow[3] = [(text[0].strip(), center)]
            else:
                closest_arrow[3].append((text[0].strip(), center))
                more_than_one_text_arrow.append(closest_arrow)
    
    for arrow in arrow_info:
        if arrow[3] is None:
            for i, m1_arrow in enumerate(more_than_one_text_arrow):
                min_distance = float("inf")
                for j, (text, center) in enumerate(m1_arrow[3]):
                    distance = abs(cv.pointPolygonTest(arrow[0], center, True))
                    if distance < min_distance:
                        min_distance = distance
                        closest_text = text
                        closest_center = center
                        closest_index = (i, j)
            arrow[3] = [(closest_text, closest_center)]
            more_than_one_text_arrow[i][3].pop(j)
            if len(more_than_one_text_arrow[i][3]) == 0:
                more_than_one_text_arrow.pop(i)
    return arrow_info, arrow_info_image