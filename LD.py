# Importing Libraries
import cv2
import numpy as np

# Creating class for Lanes detection
class Lane_Detection:
    def LD(picture = None):
        lft_lns_ftd, rgt_lns_ftd = np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])

        if picture is not None:
            # Canny edge detection
            img = picture
            minThreshold = 60
            maxThreshold = 130
            frame = cv2.Canny(img, minThreshold, maxThreshold)
            cv2.imshow('Canny Edge detection', frame)

            # Creating a mask for the image frame
            mask = Lane_Detection.mask(frame)
            cv2.imshow('Mask', mask)
            # Masking
            masked_edge_img = cv2.bitwise_and(frame, mask)
            cv2.imshow('Masked_edge_img', masked_edge_img)
            # Applying Houghlines for detecting lines
            lines = cv2.HoughLinesP(masked_edge_img, 1, np.pi / 100, 15, minLineLength=00, maxLineGap=20)

            if lines is not None:
                left_lines = [line for line in lines if -0.5 > Lane_Detection.calculate_slope(line)]
                right_lines = [line for line in lines if Lane_Detection.calculate_slope(line) > 0.5]

                # Fit the left and right lane lines separately
                lft_lns_ftd = Lane_Detection.least_squares_fit(left_lines)
                rgt_lns_ftd = Lane_Detection.least_squares_fit(right_lines)

                if lft_lns_ftd is None:
                    lft_lns_ftd = np.array([[0, 0], [0, 0]])
                if rgt_lns_ftd is None:
                    rgt_lns_ftd = np.array([[0, 0], [0, 0]])

            return lft_lns_ftd, rgt_lns_ftd

        return lft_lns_ftd, rgt_lns_ftd


    # calculating slope by two point slope formula
    def calculate_slope(line):
        x1, y1, x2, y2 = line[0]
        return (y2 - y1) / (x2 - x1 + 0.01)
    
    # Creating mask for Trepizoidal area in fron to of the vehicle
    def mask(frame):
        mask = np.zeros_like(frame)

        height = mask.shape[0]
        height_bevel = int(0.75 * height)
        length = mask.shape[1]
        length_bevel = int(0.6 * length)
        length_under = int(0.2 * length)
        mask = cv2.fillPoly(mask, np.array([[[length_under, height], [length - length_bevel, height_bevel],
                                             [length_bevel, height_bevel], [length - length_under, height],
                                             [length_under, height]]]), color=255)
        return mask

    # least square fitting
    def least_squares_fit(lines):
        x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines]) 
        y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
        if lines != None:
            if len(x_coords) >= 1:
                poly = np.polyfit(x_coords, y_coords, deg=1)
                point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
                point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
                return np.array([point_min, point_max], dtype=np.int)
            else:
                pass
        else:
            pass