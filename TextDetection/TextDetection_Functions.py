print("\nATTENTION --> Loading functions from \"TextDetection_Functions.py\":")
####################################################################################################
###################################### LIBRARIES ##################################################
####################################################################################################
print("\t- Loading file management libraries: \"os\" and \"glob\".")
import glob
import os
print("\t- Completed. Loading image and data management libraries: \"cv2\", \"csv\", \"numpy\", and \"shapely\".")
import cv2
import csv
import numpy as np
import shapely as shap
print("\t- Completed. Loading Deep Learning libraries: \"tensorflow\" and \"keras\".")
import tensorflow as tf
from tensorflow import keras
####################################################################################################
###################################### PREVIOUS STEP ##############################################
####################################################################################################
# Getting the working directory:
work_path = os.getcwd()

# Getting the training directories (images and annotations):
train_img_path = os.path.join(work_path, "Scene_Text_Datasets", "Training_Images")
train_ann_path = os.path.join(work_path, "Scene_Text_Datasets", "Training_Annotations")

# Getting the validation directories (images and annotations):
val_img_path = os.path.join(work_path, "Scene_Text_Datasets", "Validation_Images")
val_ann_path = os.path.join(work_path, "Scene_Text_Datasets", "Validation_Annotations")

####################################################################################################
################################## LOADING LANMS LIBRARY ##########################################
####################################################################################################
print("\t- Completed. Attempting to load the \"lanms\" library ...")
try:
    import lanms
    def NMS(raw_boxes, iou_threshold):
        return lanms.merge_quadrangle_n9(raw_boxes, iou_threshold)
    print("\t- Successfully loaded the \"lanms\" library and embedded it in the \"NMS\" function.")
except:
    print("\t- Failed to load the \"lanms\" library. The \"NMS\" function is loaded as a fallback.")
    def NMS(raw_boxes, iou_threshold): # Applies the "Non-Maximum Suppression" algorithm to the resulting boxes

        # Sorts the box indices based on the score and initializes a list of selected indices:
        sorted_idx = np.argsort(raw_boxes[:, 8])[::-1] # From highest to lowest "score"
        selected_idx = [] # Indices that will not be eliminated when applying the algorithm

        # Iterates through the sorted indices until there are no more indices:
        while len(sorted_idx) > 0:

            # Selects/adds the index with the highest score (the first one):
            actual_idx = sorted_idx[0]
            selected_idx.append(actual_idx)

            # Initializes the vectors that will store IoU and generates the polygon of the selected box:
            IoU_array = np.zeros(len(raw_boxes))
            selected_poly = shap.geometry.Polygon(raw_boxes[actual_idx, :8].reshape((4, 2))) # Excluding the score

            # Calculates the areas of the polygon/intersection between the selected and the rest:
            for idx in sorted_idx[1:]:

                # Generates the comparison polygon:
                other_poly = shap.geometry.Polygon(raw_boxes[idx, :8].reshape((4, 2))) # Excluding the score

                # Determines if there is an intersection, and if so, calculates the IoU:
                if selected_poly.intersects(other_poly):
                    inter_area = selected_poly.intersection(other_poly).area
                    IoU_array[idx] = inter_area /(selected_poly.area + other_poly.area - inter_area)
                else:
                    IoU_array[idx] = 0.0

            # Determines which non-selected boxes meet the threshold and can be eliminated:
            overlap_idx = np.where(IoU_array >= iou_threshold)[0]
            sorted_idx = np.delete(sorted_idx, np.where(np.in1d(sorted_idx, overlap_idx))[0])

            # To continue iterating, the already stored current index is removed:
            sorted_idx = np.delete(sorted_idx, 0)

        # At the end, you will have the list of selected indices, and it returns the independent boxes:
        return raw_boxes[selected_idx, :]
####################################################################################################
############################### FUNCTION 1: load_images_path #######################################
####################################################################################################
# DESCRIPTION: Extracts all images within an absolute "path" (imgs_path) and their associated
# annotations (anns_path). They are paired, ensuring that an image is not delivered without its
# accompanying txt file. The list of pairs is sorted according to the identification number.
def load_images_path(imgs_path, anns_path):

    # Initializes the lists that will store all absolute paths of images and annotations:
    all_img_files = []  # Images -> Generally ".jpg"
    all_ann_files = []  # Annotations -> ".txt"

    # Iterates through each image extension and retrieves the complete list of images:
    for ext in ["jpeg", "jpg", "png"]:
        all_img_files.extend(glob.glob(os.path.join(imgs_path, "*.{}".format(ext))))

    # Extracts the complete list of annotations:
    all_ann_files.extend(glob.glob(os.path.join(anns_path, "*.txt")))

    # Sorts both lists according to the identification number of the image/annotation from their paths:
    all_img_files = sorted(all_img_files.copy(), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    all_ann_files = sorted(all_ann_files.copy(), key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))

    # Initializes a list that will store image/annotation pairs:
    all_pair_files = []

    # Iterates through each image path and looks for its attached annotation:
    for file in all_img_files:

        # Generates the name of the annotation file attached to the image:
        pair_ann_file = os.path.join(anns_path, "gt_" + os.path.basename(file).split(".")[0] + ".txt")

        # Checks if it exists, and if so, generates the pair; otherwise, it is discarded:
        if pair_ann_file in all_ann_files:
            all_pair_files.append([file, pair_ann_file])

    # Completes the process by returning the list of pairs as an array:
    return np.array(all_pair_files)
####################################################################################################
############################### FUNCTION 2: maps_generator #########################################
####################################################################################################
# DESCRIPTION: Given the image dimensions as "input_sz," along with annotations (as many rows as
# bounding boxes, and as columns: ["x1","y1"], ["x2","y2"], ["x3","y3"], ["x4","y4"]), it generates
# the following maps: score map ("score_map"), geometric maps ("Top-Right-Bottom-Left-Angle"), and
# training mask ("Training_mask," only for EAST model metrics), according to the RBOX method. It uses
# the internal functions: "poly_shrinker" and "sort_rectangle."
def maps_generator(anns, labels, input_sz):

    # Internal Function 1: Responsible for "shrinking" the "Bounding Box" defined in the annotations
    def poly_shrinker(poly, r):

        # PARAMETERS:
        #   poly -> Vertices of the annotations [x1,y1], [x2,y2], [x3,y3], and [x4,y4].
        #   r -> Ratio in the original rescaled image.
        #   return -> Vertices adjusted to the rescaling.
        R = 0.3  # shrink ratio

        # Find the longer pair
        if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
            # First move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
            ## p0, p3
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
        else:
            ## p0, p3
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
        return poly

    # Internal Function 2: Finds the coordinates of the 4 vertices given by "poly" (explored clockwise)
    def sort_rectangle(poly):

        # First find the lowest point
        p_lowest = np.argmax(poly[:, 1])

        # If the bottom line is parallel to the x-axis, then p0 must be the upper-left corner
        if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
            p0_index = np.argmin(np.sum(poly, axis=1))
            p1_index = (p0_index + 1) % 4
            p2_index = (p0_index + 2) % 4
            p3_index = (p0_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], 0.0
        else:

            # Find the point that sits right to the lowest point
            p_lowest_right = (p_lowest - 1) % 4
            p_lowest_left = (p_lowest + 1) % 4
            angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))

            # This point is p2
            if angle / np.pi * 180 > 45:
                p2_index = p_lowest
                p1_index = (p2_index - 1) % 4
                p0_index = (p2_index - 2) % 4
                p3_index = (p2_index + 1) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)

            # This point is p3
            else:
                p3_index = p_lowest
                p0_index = (p3_index + 1) % 4
                p1_index = (p3_index + 2) % 4
                p2_index = (p3_index + 3) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], angle

    # (START OF MAIN) Initializes the maps and masks to be generated based on the provided dimensions:
    score_map = np.zeros((input_sz, input_sz), dtype=np.uint8)  # (height, width)
    geo_map = np.zeros((input_sz, input_sz, 5), dtype=np.float32)  # (height, width)
    vert_mask = np.zeros((input_sz, input_sz), dtype=np.uint8)  # (height, width)
    training_mask = np.ones((input_sz, input_sz), dtype=np.uint8)  # (height, width)

    # Iterates through each row of annotations and labels:
    for idx, ann_and_label in enumerate(zip(anns, labels)):

        # Initializes the ratio vector and displays the annotation/"label":
        rat = [None, None, None, None]
        BB = ann_and_label[0]
        label = ann_and_label[1]

        # Calculates the minimum of the norms/ratios and applies "poly_shrinker":
        for ii in range(len(rat)):
            rat[ii] = min(np.linalg.norm(BB[ii] - BB[(ii+1)%4]), np.linalg.norm(BB[ii] - BB[(ii-1)%4]))
        shrink_vert = poly_shrinker(BB.copy(), rat).astype(np.int32)[np.newaxis, :, :]

        # Generates the score map and vertex mask:
        cv2.fillPoly(score_map, shrink_vert, 1)
        cv2.fillPoly(vert_mask, shrink_vert, idx + 1)

        # Calculates the minimum width and height of the polygon defining the vertices:
        vert_h = min(np.linalg.norm(BB[0] - BB[3]), np.linalg.norm(BB[1] - BB[2]))
        vert_w = min(np.linalg.norm(BB[0] - BB[1]), np.linalg.norm(BB[2] - BB[3]))

        # Ignores polygons that do not reach the font size and have a label with text:
        if (min(vert_h, vert_w) < 10) or label:
            cv2.fillPoly(training_mask, BB.astype(np.int32)[np.newaxis, :, :], 0)

        # Identifies the (x, y) within the vertex mask, positions in the matrix:
        xy_mask = np.argwhere(vert_mask == (idx + 1))

        # Searches for the minimum enclosed area and finds the vertices that generate it:
        min_box = cv2.minAreaRect(BB)  # -> See https://theailearner.com/tag/cv2-minarearect/
        (p0_rect, p1_rect, p2_rect, p3_rect), angle = sort_rectangle(cv2.boxPoints(min_box))

        # NOTE: The original vertices assume a box that wraps an area, which depending
        # on its orientation can be minimal. The four corners are located on that minimal area,
        # through the analysis of that area: center, (w, h), and angle. Since the "cv2.boxPoints(box)"
        # function returns: p0_rect, p1_rect, p2_rect, and p3_rect.

        # Iterates through the (x, y) of the vertices to extract the geometric maps:
        for y, x in xy_mask:
            point = np.array([x, y], dtype=np.float32)
            geo_map[y, x, 0] = np.linalg.norm(np.cross(p1_rect - p0_rect, p0_rect - point))/np.linalg.norm(p1_rect - p0_rect)  # Top
            geo_map[y, x, 1] = np.linalg.norm(np.cross(p2_rect - p1_rect, p1_rect - point))/np.linalg.norm(p2_rect - p1_rect)  # Right
            geo_map[y, x, 2] = np.linalg.norm(np.cross(p3_rect - p2_rect, p2_rect - point))/np.linalg.norm(p3_rect - p2_rect)  # Bottom
            geo_map[y, x, 3] = np.linalg.norm(np.cross(p0_rect - p3_rect, p3_rect - point))/np.linalg.norm(p0_rect - p3_rect)  # Left
            geo_map[y, x, 4] = angle  # Angle

    # Returns the maps:
    return score_map, geo_map, training_mask
####################################################################################################
############################### FUNCTION 3: box_regeneration #######################################
####################################################################################################
# DESCRIPTION: It assumes the reverse option of "maps_generator." It receives the following maps:
# score map ("score_map") and geometric maps ("Top-Right-Bottom-Left-Angle") returned by the model, and
# the ratio (ratio in "width", ratio in "height") of resizing the image that was introduced into the
# model. This function retrieves the "Bounding Boxes" from the maps and adapts them to the
# original image (respecting the ratios). The output of the function will be a vector of dimensions (X,4,2),
# where X is the number of different annotations (4 and 2 for the pairs of components that define the
# vertices: []"x1","y1"],["x2","y2"],["x3","y3"],["x4","y4"]). It depends on an internal function:
# "restore_rectangle" and a special package: "lamns".
def box_regeneration(score_map, geo_map, ratios):

    # Internal Function 1: Obtains the vertex vector through the score and geometric maps
    def restore_rectangle(orig, geomt):

        # Extracts "Top-Right-Bottom-Left" and "angle" from the geometric map:
        TRDL = geomt[:, :4]
        angle = geomt[:, 4]

        # Adapts for the case of an angle greater than 0 (filtered):
        orig_0 = orig[angle >= 0]
        TRDL_0 = TRDL[angle >= 0]
        angle_0 = angle[angle >= 0]

        # In the case that the angle is greater than 0, it is applied:
        if orig_0.shape[0] > 0:

            # Generates the matrix that condenses the geometric structure of the box:
            p = np.array([np.zeros(TRDL_0.shape[0]), -TRDL_0[:, 0] - TRDL_0[:, 2], TRDL_0[:, 1] + TRDL_0[:, 3], -TRDL_0[:, 0] - TRDL_0[:, 2],
                         TRDL_0[:, 1] + TRDL_0[:, 3], np.zeros(TRDL_0.shape[0]), np.zeros(TRDL_0.shape[0]), np.zeros(TRDL_0.shape[0]), TRDL_0[:, 3],
                         -TRDL_0[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2)) #-> N*5*2

            # Determines the rotation matrices to be applied to the box in the x and y axes:
            rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  #-> N*5*2
            rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            # Rotates the generated geometric matrix:
            p_rotate_x = np.sum(rotate_matrix_x*p, axis=2)[:, :, np.newaxis]  #-> N*5*1
            p_rotate_y = np.sum(rotate_matrix_y*p, axis=2)[:, :, np.newaxis]  #-> N*5*1
            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  #-> N*5*2

            # Determines the 4 vertices of the polygon:
            p3_in_origin = orig_0 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  #-> N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin
            new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :], new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  #-> N*4*2
        else:
            new_p_0 = np.zeros((0, 4, 2)) # Case where the angle is not negative

        # In the case of a negative angle, the other point of the polygon is initialized:
        orig_1 = orig[angle < 0]
        TRDL_1 = TRDL[angle < 0]
        angle_1 = angle[angle < 0]

        # In the case that the angle is less than 0, it is applied:
        if orig_1.shape[0] > 0:

            # Generates the matrix that condenses the geometric structure of the box (now negative angle):
            p = np.array([-TRDL_1[:, 1] - TRDL_1[:, 3], -TRDL_1[:, 0] - TRDL_1[:, 2], np.zeros(TRDL_1.shape[0]), -TRDL_1[:, 0] - TRDL_1[:, 2],
                          np.zeros(TRDL_1.shape[0]), np.zeros(TRDL_1.shape[0]), -TRDL_1[:, 1] - TRDL_1[:, 3], np.zeros(TRDL_1.shape[0]), -TRDL_1[:, 1], -TRDL_1[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  #-> N*5*2

            # Determines the rotation matrices to be applied to the box in x and y axes (now negative angle):
            rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1)) #-> N*5*2
            rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            # Rotates the generated geometric matrix (now negative angle):
            p_rotate_x = np.sum(rotate_matrix_x*p, axis=2)[:, :, np.newaxis]  #-> N*5*1
            p_rotate_y = np.sum(rotate_matrix_y*p, axis=2)[:, :, np.newaxis]  #-> N*5*1
            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  #-> N*5*2

            # Determines the 4 vertices of the polygon (now negative angle):
            p3_in_origin = orig_1 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  #-> N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin
            new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :], new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  #-> N*4*2
        else:
            new_p_1 = np.zeros((0, 4, 2)) # Case where the angle is not negative

        # Returns the restoration of the text box:
        return np.concatenate([new_p_0, new_p_1])

    # (START OF MAIN) Filtering of the score map by "threshold" and the boxes are sorted by the y-axis:
    xy_text = np.argwhere(score_map > 0.8)
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    # The boxes are restored from the maps and adapted to the appropriate format:
    box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  #-> N*4*2
    boxes = np.zeros((box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    # NOTE: The internal function "restore_rectangle" manages to generate the "bounding boxes" from the
    # geometric and score maps to a format (X,4,2), that is: X boxes, 4 vertices of 2 coordinates
    # (x,y). Subsequently, the format must be changed to (X, 9), that is, X boxes and 8 components
    # (4 vertices x 2 -> x1,y1,x2,y2,x3,y3,x4,y4) plus one more component for the score value it has.
    # this format is what the "lanms.merge_quadrangle_n9" function receives.

    # The NMS algorithm is applied to group the detected boxes:
    boxes = NMS(boxes.astype("float32"), 0.2)

    # NOTE: Since the "lanms" package causes problems, an internal function known as
    # "NMS" will be created to replace it, although it is not the same algorithm. Its 
    # mission will be to reduce the number of boxes remaining and return the remaining boxes in
    # format (X', 9). Subsequently, another filtering will be applied and it will finally return
    # to format (X',4,2). Being X'>X.

    # Checks if there are boxes or not:
    if boxes.shape[0] == 0:
        boxes = []
    else:

        # Here some low-score boxes are filtered by the average score map (different from the original EAST):
        for ii, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32)//4, 1)
            boxes[ii, 8] = cv2.mean(score_map, mask)[0]

        # The final filtering is done by box detection "threshold":
        boxes = boxes[boxes[:, 8] > 0.1]

        # Adjust their final dimensions to a vector [x1,y1],...,[x4,y4] and to the resizing ratio:
        boxes = boxes[:, :8].reshape(-1, 4, 2)
        boxes[:, :, 0] /= ratios[0] #Ratio in "width"
        boxes[:, :, 1] /= ratios[1] #Ratio in "height"
        boxes = boxes.astype(np.int32)

    # Return the detected boxes:
    return boxes
####################################################################################################
################################# FUNCTION 4: metrics ##############################################
####################################################################################################
# DESCRIPTION: It is responsible for calculating the metrics of "Recall," "Precision," and "F-score" for an image.
# To do this, it receives the annotations of the true "Ground Truth" Bounding Boxes and the predicted ones
# in list format. It requires the internal function: "IOU_calculation" and assumes that there is no conflict
# of the same IoU for multiple predictions. The following sources are used:
# -> https://github.com/matterport/Mask_RCNN/issues/2513
# -> https://blog.zenggyu.com/posts/en/2018-12-16-an-introduction-to-evaluation-metrics-for-object-detection/index.html
# -> https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
def metrics(pred_anns, gt_anns):

    # Internal Function 1: Calculation of Intersection over Union (IoU) of the predicted Bounding Box
    def IOU_calculation(BB_pred, all_BB_gt):

        # Initialize the vector that will store the IoU of each Ground truth:
        IoU_array = np.empty((1, len(all_BB_gt)))

        # Iterate over each of the true annotations ("Ground Truth"):
        for idx, gt_BB in enumerate(all_BB_gt):

            # Defines the polygon that arises from joining the vertices (clockwise):
            gt_poly = shap.geometry.Polygon(gt_BB)

            # Calculates the intersection area between the predicted polygon and the "Ground Truth":
            if BB_pred.intersects(gt_poly):
                inter_area = BB_pred.intersection(gt_poly).area
            else:
                inter_area = 0.0

            # Calculates the IoU and stores it in the vector:
            IoU_array[0, idx] = inter_area/(BB_pred.area + gt_poly.area - inter_area)

        # Returns the IoU calculation per "Ground Truth Bounding Boxes":
        return IoU_array

    # (START OF MAIN) Determines if there is really a prediction:
    if len(pred_anns) != 0:

        # Initialize the vector that will store the calculated IoU:
        IoU_table = np.empty((0, len(gt_anns)))

        # Iterate over each of the prediction Bounding Boxes:
        for pred_BB in pred_anns:

            # Defines the prediction polygon with the vertices:
            pred_poly = shap.geometry.Polygon(pred_BB)

            # Calculates IoU and stores it inside the generated table:
            IoU_table = np.vstack((IoU_table, IOU_calculation(pred_poly, gt_anns)))

        # Finds the maximum IoU of each row that exceeds the "Threshold" of 0.5:
        IoU_table[IoU_table < 0.5] = 0.0
        Max_IoU = np.nanmax(IoU_table, axis=1)

        # Counts how many predictions are "True Positives," how many are "False Positives," and how many true ones are "False Negatives";
        TP = np.sum(Max_IoU != 0)
        FP = IoU_table.shape[0] - TP
        FN = IoU_table.shape[1] - TP

        # Calculate the metrics of "Recall," "Precision," and "F-score":
        recall = TP/(TP + FN)
        precision = TP/(TP + FP)
        if (precision + recall) != 0:
            fscore = 2*(precision*recall)/(precision + recall)
        else:
            fscore = 0.0

        # Returns the metrics in a dictionary:
        return {"IoU":IoU_table, "recall":recall*100, "precision":precision*100, "fscore":fscore*100}

    # In case there are no predictions:
    return {"IoU":np.array([]), "recall":0, "precision":0, "fscore":0}
####################################################################################################
########################### FUNCTION 5: load_inference_image #######################################
####################################################################################################
# DESCRIPTION: It is responsible for loading the image and paired annotation in the absolute "path."
# These paths correspond to the validation images to which metrics can be applied. The images must
# be resized to dimensions, multiples of 32, which allows their input into the EAST neural network model
# (This is done in the internal function "load_image_and_resize" and returns the scaled image,
# the original dimensions (height, width), and the scaling ratios in the same order). The annotations
# must be corrected to the same ratios (Internal function: load_and_correct_annotations").
def load_inference_image(img_and_ann_path):

    # Function 1: Loads the image given by the absolute "path" and resizes it
    def load_image_and_resize(img_path):

        # Load the image and its dimensions:
        im = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB
        height, width, _ = im.shape

        # Calculate the lateral limitation ratio according to the maximum permissible:
        if max(height, width) > 2400:
            ratio = (2400/height if height > width else 2400/width)
        else:
            ratio = 1.0

        # Correct the sides so that they do not exceed the limitation ratio:
        height_rs = int(height*ratio)
        width_rs = int(width*ratio)

        # Limit these dimensions to multiples of 32:
        height_rs = height_rs if height_rs % 32 == 0 else (height_rs // 32 - 1) * 32
        width_rs = width_rs if width_rs % 32 == 0 else (width_rs // 32 - 1) * 32
        height_rs = max(32, height_rs)
        width_rs = max(32, width_rs)

        # Calculate the final resizing ratios:
        h_rat = height_rs/float(height)
        w_rat = width_rs/float(width)

        # Return the scaled image, along with the original dimensions and scaling ratios:
        return cv2.resize(im, (int(width_rs), int(height_rs))), (height, width), (h_rat, w_rat)

    # Internal Function 2: Read the TXT according to the absolute "path" of the annotations and correct them
    def load_and_correct_annotations(ann_path, orig_size, resize_rat):

        # Initialize the lists where the vertices of the "Bounding Box" will go:
        BB_list = []

        # Open the file, pass it through the CSV reader, and analyze line by line:
        with open(ann_path, "r") as file_opener:
            reader = csv.reader(file_opener)
            for line in reader:

                # Remove BOM from each line and segment it into elements:
                line = [word.strip("\ufeff").strip("\xef\xbb\xbf") for word in line]

                # If the number of elements is more than 9, remove the remaining ones:
                if len(line) > 9:
                    line = line[:9]

                # Extract the coordinates of the 4 corners of the "Bounding Box" and store them:
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
                BB_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        # Construct the annotations as an "array" and extract the dimensions of the original image:
        BB_array = np.array(BB_list, dtype=np.float32)
        height, width = orig_size

        # Validate that the coordinates (x, y) are within the image (width x height):
        BB_array[:, :, 0] = np.clip(BB_array[:, :, 0], 0, width - 1)  # X
        BB_array[:, :, 1] = np.clip(BB_array[:, :, 1], 0, height - 1)  # Y

        # Calculate the area of the resulting polygon for each of the annotations:
        area = np.sum([(BB_array[:, 1, 0] - BB_array[:, 0, 0]) * (BB_array[:, 1, 1] + BB_array[:, 0, 1]),
                       (BB_array[:, 2, 0] - BB_array[:, 1, 0]) * (BB_array[:, 2, 1] + BB_array[:, 1, 1]),
                       (BB_array[:, 3, 0] - BB_array[:, 2, 0]) * (BB_array[:, 3, 1] + BB_array[:, 2, 1]),
                       (BB_array[:, 0, 0] - BB_array[:, 3, 0]) * (BB_array[:, 0, 1] + BB_array[:, 3, 1])], axis=0)/2.0

        # Analyze one by one the annotations to modify them:
        for ii in range(len(area)):

            # Remove annotations with small area:
            if abs(area[ii]) < 1:
                BB_array = np.delete(BB_array, ii, axis=0)

            # Change the orientation to "clockwise" if the area is positive:
            if area[ii] > 0:
                BB_array[ii, :, :] = BB_array[ii, (0, 3, 2, 1), :]

        # Resize the annotations to the input dimensions respecting the ratios:
        BB_array[:, :, 0] *= resize_rat[1]  # Width
        BB_array[:, :, 1] *= resize_rat[0]  # Height

        # Return the corrected and scaled annotations:
        return BB_array

    # (START OF MAIN) Load the resized image and annotations contained in the attached txt:
    img_resize, orig_dim, ratios = load_image_and_resize(img_and_ann_path[0])  # Order is (height, width)
    ann_xy = load_and_correct_annotations(img_and_ann_path[1], orig_dim, ratios)

    # NOTE: The structure of the annotations is an array of dimensions (X, Y, Z), where X is the number
    # of different annotations (each "Bounding Box"), Y is each of the vertices that defines a
    # "Bounding Box" (there are 4), and Z is whether the x coordinate (0) or y coordinate (1) of the vertex.

    # Return the scaled image and corrected annotations:
    return img_resize.astype(np.float32), ann_xy
####################################################################################################
###################### FUNCTION 6: load_non_augmententation_images #################################
####################################################################################################
# DESCRIPTION: It is responsible for loading the images and paired annotations in the absolute "path."
# Later, it must correct the annotations and generate the maps of the image, to group them in
# lists, all scaled to the square dimensions that the model must receive (input_size).
# It uses the internal functions: "correct_annotations" and "correct_and_resize".
def load_non_augmententation_images(imgs_and_anns_paths, input_size):

    # Internal Function 1: Read the TXT according to the absolute "path" of the annotations
    def load_annotations(ann_path):

        # Initialize the lists where the vertices of the "Bounding Box" and its "label" will go:
        BB_list = []
        label = []

        # Open the file, pass it through the CSV reader, and analyze line by line:
        with open(ann_path, "r") as file_opener:
            reader = csv.reader(file_opener)
            for line in reader:

                # Remove BOM from each line and segment it into elements:
                line = [word.strip("\ufeff").strip("\xef\xbb\xbf") for word in line]

                # If the number of elements is more than 9, remove the remaining ones:
                if len(line) > 9:
                    line = line[:9]

                # Extract the coordinates of the 4 corners of the "Bounding Box" and store them:
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
                BB_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

                # Extract the "label" (last element) and filter according to whether it is identified or not:
                if line[-1] == "###":
                    label.append(True)
                else:
                    label.append(False)

        # Return the extracted annotations in "array" format:
        return np.array(BB_list, dtype=np.float32), np.array(label, dtype=np.bool_)

    # Internal Function 2: Correction and resizing of the image and annotations to the input dimensions
    def correct_and_resize(orig_im, orig_ann, orig_label, input_sz):

        # Extract the dimensions of the original image:
        height, width, _ = orig_im.shape

        # Validate that the coordinates (x, y) are within the image (width x height):
        orig_ann[:, :, 0] = np.clip(orig_ann[:, :, 0], 0, width - 1)  #X
        orig_ann[:, :, 1] = np.clip(orig_ann[:, :, 1], 0, height - 1)  #Y

        # Calculate the area of the resulting polygon for each of the annotations:
        area = np.sum([(orig_ann[:, 1, 0] - orig_ann[:, 0, 0]) * (orig_ann[:, 1, 1] + orig_ann[:, 0, 1]),
                       (orig_ann[:, 2, 0] - orig_ann[:, 1, 0]) * (orig_ann[:, 2, 1] + orig_ann[:, 1, 1]),
                       (orig_ann[:, 3, 0] - orig_ann[:, 2, 0]) * (orig_ann[:, 3, 1] + orig_ann[:, 2, 1]),
                       (orig_ann[:, 0, 0] - orig_ann[:, 3, 0]) * (orig_ann[:, 0, 1] + orig_ann[:, 3, 1])], axis=0)/2.0

        # Analyze one by one the annotations to modify them and their "label":
        for ii in range(len(area)):

            # Remove annotations/"label" with small area:
            if abs(area[ii]) < 1:
                orig_ann = np.delete(orig_ann, ii, axis=0)
                orig_label = np.delete(orig_label, ii, axis=0)

            # Change the orientation to "clockwise" if the area is positive:
            if area[ii] > 0:
                orig_ann[ii, :, :] = orig_ann[ii, (0, 3, 2, 1), :]

        # Resize the annotations to the input dimensions respecting the ratios:
        orig_ann[:, :, 0] *= input_sz/width  # Width
        orig_ann[:, :, 1] *= input_sz/height  # Height

        # Return the corrected annotations and the resized image:
        return cv2.resize(orig_im, dsize=(input_sz, input_sz)), orig_ann, orig_label

    # (START OF MAIN) Initialize empty vectors of images and maps to generate:
    images =  np.empty((0, input_size, input_size, 3))
    score_maps = np.empty((0, int(input_size/4), int(input_size/4), 1))
    geo_maps = np.empty((0, int(input_size/4), int(input_size/4), 5))
    training_masks = np.empty((0, int(input_size/4), int(input_size/4), 1))

    # Iterate over each pair of image and annotation file paths:
    for pair_path in imgs_and_anns_paths:

        # Read the image and annotations contained in the attached txt:
        img = cv2.imread(pair_path[0])[:, :, ::-1]  # BGR -> RGB
        ann_xy, ann_label = load_annotations(pair_path[1])

        # NOTE: Since the image has been read with cv2, it returns the image in BGR, but it should
        # be displayed as RGB. As for the structure of the annotations, it is a vector of dimensions
        # (X, Y, Z), where X is the number of different annotations (each "Bounding Box"), Y is each
        # of the vertices that defines a "Bounding Box" (there are 4), and Z is whether the x coordinate
        # (0) or y coordinate (1) of the vertex.

        # Resize the image and annotations to the model's input dimensions:
        img_rs, ann_xy_rs, ann_label_rs = correct_and_resize(img, ann_xy, ann_label, input_size)

        # Generate score and geometric maps:
        score_map, geo_map, training_mask = maps_generator(ann_xy_rs, ann_label_rs, input_size)

        # Store the images and generated maps in the appropriate format:
        images = np.vstack((images, (img_rs[np.newaxis, :].astype(np.float32))))
        score_maps = np.vstack((score_maps, score_map[np.newaxis, ::4, ::4, np.newaxis].astype(np.float32)))
        geo_maps = np.vstack((geo_maps, geo_map[np.newaxis, ::4, ::4, :].astype(np.float32)))
        training_masks = np.vstack((training_masks, training_mask[np.newaxis, ::4, ::4, np.newaxis].astype(np.float32)))

        # NOTE: As for the maps, they are reduced to one-fourth by taking one component every
        # 4 and those maps that are of a single channel have an extra axis added so that they are in the
        # same tensor type.

    return images, score_maps, geo_maps, training_masks
####################################################################################################
###################### FUNCTION 7: load_augmententation_images ##########################################
####################################################################################################
# DESCRIPTION: It is responsible for loading the images and paired annotations in the absolute "path."
# Later, it must correct the annotations and generate the maps of the image, to group them in
# lists, all scaled to the square dimensions that the model must receive (input_size) and
# applying data augmentation. It uses the internal functions: "correct_annotations", "correct_and_resize"
# and "crop_area".
def load_augmententation_images(imgs_and_anns_paths, input_size):

    # Internal Function 1: Read the TXT according to the absolute "path" of the annotations
    def load_annotations(ann_path):

        # Initialize the lists where the vertices of the "Bounding Box" and its "label" will go:
        BB_list = []
        label = []

        # Open the file, pass it through the CSV reader, and analyze line by line:
        with open(ann_path, "r") as file_opener:
            reader = csv.reader(file_opener)
            for line in reader:

                # Remove BOM from each line and segment it into elements:
                line = [word.strip("\ufeff").strip("\xef\xbb\xbf") for word in line]

                # If the number of elements is more than 9, remove the remaining ones:
                if len(line) > 9:
                    line = line[:9]

                # Extract the coordinates of the 4 corners of the "Bounding Box" and store them:
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
                BB_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

                # Extract the "label" (last element) and filter according to whether it is identified or not:
                if line[-1] == "###":
                     label.append(True)
                else:
                    label.append(False)

        # Return the extracted annotations in "array" format:
        return np.array(BB_list, dtype=np.float32), np.array(label, dtype=np.bool_)

    # Internal Function 2: Correction of annotations to the original dimensions of the image
    def correct_annotations(orig_im, orig_ann, orig_label):

        # Extract the dimensions of the original image:
        height, width, _ = orig_im.shape

        # Validate that the coordinates (x, y) are within the image (width x height):
        orig_ann[:, :, 0] = np.clip(orig_ann[:, :, 0], 0, width - 1)  # X
        orig_ann[:, :, 1] = np.clip(orig_ann[:, :, 1], 0, height - 1)  # Y

        # Calculate the area of the resulting polygon for each of the annotations:
        area = np.sum([(orig_ann[:, 1, 0] - orig_ann[:, 0, 0]) * (orig_ann[:, 1, 1] + orig_ann[:, 0, 1]),
                       (orig_ann[:, 2, 0] - orig_ann[:, 1, 0]) * (orig_ann[:, 2, 1] + orig_ann[:, 1, 1]),
                       (orig_ann[:, 3, 0] - orig_ann[:, 2, 0]) * (orig_ann[:, 3, 1] + orig_ann[:, 2, 1]),
                       (orig_ann[:, 0, 0] - orig_ann[:, 3, 0]) * (orig_ann[:, 0, 1] + orig_ann[:, 3, 1])], axis=0)/2.0

        # Analyze one by one the annotations to modify them and their "label":
        for ii in range(len(area)):

            # Remove annotations/"label" with small area:
            if abs(area[ii]) < 1:
                orig_ann = np.delete(orig_ann, ii, axis=0)
                orig_label = np.delete(orig_label, ii, axis=0)

            # Change the orientation to "clockwise" if the area is positive:
            if area[ii] > 0:
                orig_ann[ii, :, :] = orig_ann[ii, (0, 3, 2, 1), :]

        # Return the corrected annotations to original dimensions:
        return orig_ann, orig_label

    # Internal Function 3: Random area crop of the image
    def crop_area(im, anns, labels, crop_flag):

        # Extract the dimensions of the image and the crop mask:
        height, width, _ = im.shape
        pad_h = height//10
        pad_w = width//10

        # Initialize the crop masks:
        h_array = np.zeros((height + pad_h*2), dtype=np.int32)
        w_array = np.zeros((width + pad_w*2), dtype=np.int32)

        # Iterate over each text annotation of the image:
        for BB in anns:

            # Round the annotations and complete the crop masks:
            BB = np.round(BB, decimals=0).astype(np.int32)
            h_array[(np.min(BB[:, 1]) + pad_h):(np.max(BB[:, 1]) + pad_h)]
            w_array[(np.min(BB[:, 0]) + pad_w):(np.max(BB[:, 0]) + pad_w)]

        # Check if the crop area is too small and if so, return everything without cropping:
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        if len(h_axis) == 0 or len(w_axis) == 0:
            return im, anns, labels

        # If it meets the conditions, proceed to make several attempts to crop it:
        for intent in range(50):  # 50 attempts

            # Calculate a selection of random points from the crop masks:
            x_select = np.random.choice(w_axis, size=2)
            y_select = np.random.choice(h_axis, size=2)

            # Find min/max within the dimensions in x:
            xmin = np.clip((np.min(x_select) - pad_w), 0, width - 1)
            xmax = np.clip((np.max(x_select) - pad_w), 0, width - 1)

            # Find min/max within the dimensions in y:
            ymin = np.clip((np.min(y_select) - pad_h), 0, height - 1)
            ymax = np.clip((np.max(y_select) - pad_h), 0, height - 1)

            # Check if it is not too small and retry:
            if (xmax - xmin < 0.1*width) or (ymax - ymin < 0.1*height):
                continue

            # Check if there are any annotations within the cropping limits:
            if anns.shape[0] != 0:
                anns_in_area = (anns[:, :, 0] >= xmin) & (anns[:, :, 0] <= xmax) & (anns[:, :, 1] >= ymin) & (anns[:, :, 1] <= ymax)
                select_anns = np.where(np.sum(anns_in_area, axis=1) == 4)[0]  #Select indices of annotations within the limits
            else:
                select_anns = np.empty((0,))  #If there are no annotations, have an empty list

            # If there are no annotations within the cropping area:
            if len(select_anns) == 0:

                # Decide whether to crop or not and skip:
                if crop_flag:
                    return im[ymin : ymax + 1, xmin : xmax + 1, :], anns[select_anns], labels[select_anns]
                else:
                    continue

            # If there are annotations, adjust for cropping (leaving those selected) and return:
            anns = anns[select_anns]
            labels = labels[select_anns]
            anns[:, :, 0] -= xmin
            anns[:, :, 1] -= ymin
            return im[ymin : ymax + 1, xmin : xmax + 1, :], anns, labels

        # If it cannot be done in 50 attempts, leave it as it is:
        return im, anns, labels

    # (START OF MAIN) Initialize empty vectors of images and maps to be generated:
    images =  np.empty((0, input_size, input_size, 3))
    score_maps = np.empty((0, int(input_size/4), int(input_size/4), 1))
    geo_maps = np.empty((0, int(input_size/4), int(input_size/4), 5))
    training_masks = np.empty((0, int(input_size/4), int(input_size/4), 1))

    # Iterate over each pair of file paths of image and annotation:
    for pair_path in imgs_and_anns_paths:

        # Read the image and annotations contained in the attached txt:
        img = cv2.imread(pair_path[0])[:, :, ::-1]  # BGR -> RGB
        ann_xy, ann_label = load_annotations(pair_path[1])

        # NOTE: Since the image has been read with cv2, it returns the image in BGR, but it should actually
        # be displayed as RGB. Regarding the structure of the annotations, it is a vector with dimensions
        # (X, Y, Z), where X is the number of different annotations (each "Bounding Box"), Y is each one
        # of the vertices that defines a "Bounding Box" (there are 4), and Z is whether the x-coordinate
        # (0) or y-coordinate (1) of the vertex.

        # Correct the annotations to the original dimensions of the image:
        ann_xy, ann_label = correct_annotations(img, ann_xy, ann_label)

        # (DATA AUGMENTATION) Choose a random scale for the image and scale the image:
        rd_scale = np.random.choice(np.array([0.5, 1, 2.0, 3.0]))
        img = cv2.resize(img, dsize=None, fx=rd_scale, fy=rd_scale)

        # Correct the dimensions of the annotations to the new scale:
        ann_xy *= rd_scale  # Width and Height

        # Make a random crop of the image area:
        if np.random.rand() < 3/8:
            img, ann_xy, ann_label = crop_area(img, ann_xy, ann_label, True)

            # If there are still annotations -> "cannot find background" and skip the image:
            if ann_xy.shape[0] > 0:
                continue

            # Extract the dimensions of the new image and compare with the input dimension:
            new_h, new_w, _ = img.shape
            max_hw_input = np.max([new_h, new_w, input_size])

            # Fill with 0 the part of the image that does not reach the input dimension:
            im_padded = np.zeros((max_hw_input, max_hw_input, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = img.copy()

            # Adjust the image to the input dimensions and generate empty maps (0 annotations):
            img = cv2.resize(im_padded, dsize=(input_size, input_size))
            score_map = np.zeros((input_size, input_size), dtype=np.uint8)
            geo_map = np.zeros((input_size, input_size, 5), dtype=np.float32)
            training_mask = np.ones((input_size, input_size), dtype=np.uint8)

        # If it does not meet the "background" conditions of the crop, crop with annotations:
        else:
            img, ann_xy, ann_label = crop_area(img, ann_xy, ann_label, False)

            # If there are no annotations, skip the image:
            if ann_xy.shape[0] == 0:
                continue

            # Extract the dimensions of the new image and compare with the input dimension:
            new_h, new_w, _ = img.shape
            max_hw_input = np.max([new_h, new_w, input_size])

            # Fill with 0 the part of the image that does not reach the input dimension:
            im_padded = np.zeros((max_hw_input, max_hw_input, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = img.copy()

            # Generate the new image and extract its modified dimensions:
            img = im_padded
            new_h, new_w, _ = img.shape

            # Resize the image and the list of annotations to the new dimensions:
            img = cv2.resize(img, dsize=(input_size, input_size))
            ann_xy[:, :, 0] *= input_size/float(new_w)  # Width
            ann_xy[:, :, 1] *= input_size/float(new_h)  # Height

            # Generate score and geometric maps:
            score_map, geo_map, training_mask = maps_generator(ann_xy, ann_label, input_size)

        # Store the generated images and maps in the appropriate format:
        images = np.vstack((images, (img[np.newaxis, :].astype(np.float32))))
        score_maps = np.vstack((score_maps, score_map[np.newaxis, ::4, ::4, np.newaxis].astype(np.float32)))
        geo_maps = np.vstack((geo_maps, geo_map[np.newaxis, ::4, ::4, :].astype(np.float32)))
        training_masks = np.vstack((training_masks, training_mask[np.newaxis, ::4, ::4, np.newaxis].astype(np.float32)))

        # NOTE: Regarding the maps, they are reduced to one-fourth by taking one component every
        # 4, and those maps that are single-channel have an extra axis added to match the tensor type.

    # Return all the list of images and maps:
    return images, score_maps, geo_maps, training_masks
####################################################################################################
############################# FUNCTION 8: loss_function ############################################
####################################################################################################
# DESCRIPTION: It calculates the loss value to be applied to the training of the neural network.
# It calculates the loss of the score and geometric map according to the official EAST paper.
# It requires an internal function: "dice_coeff".
def loss_function(y_true, y_pred):

    # Internal Function 1: Calculation of the "Dice coefficient":
    def dic_coeff(y_true_score, y_pred_score, training_mask):
        intersection = tf.reduce_sum(y_true_score * y_pred_score * training_mask)
        union = tf.reduce_sum(y_true_score) + tf.reduce_sum(y_pred_score) + 1e-5
        return (1.0 - (2*intersection/union))

    # Extract training mask and true output:
    training_mask = y_true[:, :, :, -1:]
    y_true = y_true[:, :, :, :-1]

    # Calculate "Dice" classification loss and report:
    classification_loss = 0.01 * dic_coeff(y_true[:, :, :, -1:], y_pred[:, :, :, -1:], training_mask)
    tf.summary.scalar("classification_dice_loss", classification_loss)

    # Obtain true/predicted sides/angles:
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true[:, :, :, :-1], num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred[:, :, :, :-1], num_or_size_splits=5, axis=3)

    # Calculate contained area true/predicted:
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)

    # Calculate the intersection of both areas:
    w_intersect = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_intersect = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_intersect * h_intersect
    area_union = area_gt + area_pred - area_intersect

    # Calculate the "loss" and return the corrected value:
    Lg = (-tf.math.log((area_intersect + 1.0) / (area_union + 1.0))) + (20 * (1 - tf.math.cos(theta_pred - theta_gt)))
    return (300 * (tf.reduce_mean(Lg * y_true[:, :, :, -1:] * training_mask) + classification_loss))
####################################################################################################
################################ CLASS 1: TrainingSequence ########################################
####################################################################################################
# DESCRIPTION: Used by the model trainer to load new images for the model and group them into "batches."
# The structure follows the typical format indicated by Keras.
class TrainingSequence(tf.keras.utils.Sequence):
    def __init__(self):
        self.filenames = load_images_path(train_img_path, train_ann_path)
        self.batch_size = 4
        self.index = np.array(range(len(self.filenames)))
        np.random.shuffle(self.index)

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        filenames = self.filenames[self.index[idx * self.batch_size: (idx + 1) * self.batch_size]]
        images, score_maps, geo_maps, training_masks = load_augmententation_images(filenames, 512)
        if len(images) == 0:
            return self.__getitem__((idx + 1) % self.__len__())
        return tf.convert_to_tensor(images), tf.convert_to_tensor(np.concatenate((geo_maps, score_maps, training_masks), axis=-1))
        # -> Number of Images x (input_size x input_size x channels)
        # -> Number of Images x [(geo_size -> X,X,5) x (score_size -> X,X,1) x (training_size -> X,X,1)]
####################################################################################################
################################ CLASS 2: ValidationSequence #######################################
####################################################################################################
# DESCRIPTION: Used by the model validator to load new images for the model and group them into "batches."
# The structure follows the typical format indicated by Keras (identical to "TrainingSequence"), but without
# data augmentation, as inference does not require augmentation.
class ValidationSequence(tf.keras.utils.Sequence):
    def __init__(self):
        self.filenames = load_images_path(val_img_path, val_ann_path)
        self.batch_size = 4
        self.index = np.array(range(len(self.filenames)))
        np.random.shuffle(self.index)

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        filenames = self.filenames[self.index[idx * self.batch_size: (idx + 1) * self.batch_size]]
        images, score_maps, geo_maps, training_masks = load_non_augmententation_images(filenames, 512)
        if len(images) == 0:
            return self.__getitem__((idx + 1) % self.__len__())
        return tf.convert_to_tensor(images), tf.convert_to_tensor(np.concatenate((geo_maps, score_maps, training_masks), axis=-1))
        # -> Number of Images x (input_size x input_size x channels)
        # -> Number of Images x [(geo_size -> X,X,5) x (score_size -> X,X,1) x (training_size -> X,X,1)]
####################################################################################################
############################## MODEL 1: model_assembler ############################################
####################################################################################################
# DESCRIPTION: Creates the model to be used for training. It only generates the EAST-based model,
# but with "MobileNet".

def model_assembler():

    # Load the input layer along with the input preprocessing layer:
    input_ly = keras.Input(shape=[None, None, 3], dtype=tf.float32, name="Input_LY")
    input_ly = keras.applications.resnet.preprocess_input(input_ly)

    # NOTE (Original): RGB -> BGR conversion and subtract the means of the ImageNet dataset, but note
    # that although cv2 loads as BGR, the generator has changed to RGB.

    # Load the base model coupled with the input layer and initialize the list of layers:
    base_model = keras.applications.MobileNet(input_tensor=input_ly, include_top=False)
    base_model_layers = []  # Only the layers to be used will be stored here

    # Iterate through all the layers of the base model:
    for layer in base_model.layers:

        # Allow the layer to be trainable:
        layer.trainable = True

        # Store the layers to be used:
        if layer.name in ["conv_pw_2_relu", "conv_pw_5_relu", "conv_pw_11_relu", "conv_pw_13_relu"]:
            base_model_layers.append(layer)

    # Initialize the output flows of EAST:
    base_model_layers.reverse()  # From bottom to top
    f = [layer.output for layer in base_model_layers]  # Feature maps
    h = [None, None, None, None]  # Merged feature map
    g = [None, None, None, None]  # Merge base
    outputs_size = [None, 128, 64, 32]

    # NOTE: The EAST structure has two important blocks: "Feature extractor stem" (formerly PVANet and now
    # MobileNet) and "Feature-merging branch". The outputs of the first block are the "f" and those of the second are identified
    # with the "h" that has the dimensions "outputs_size". In the case of both blocks, the numbering goes from bottom to top,
    # so the layers of MobileNet are reordered with "reverse". The "g" are the "merge base" that receives the
    # outputs "h" (an output of type "h" is received by a layer "g").

    # Iterate through each of the states of both blocks of EAST:
    for idx in range(len(f)):

        # In the case of 0, both outputs coincide (f1 = h1):
        if idx == 0:
            h[idx] = f[idx]

        # Otherwise, "h" is defined as Conv3x3(Conv1x1(...)):
        else:
            conv_1x1 = keras.layers.Conv2D(filters=outputs_size[idx], kernel_size=1, activation="relu", padding="same",
                                           kernel_regularizer=keras.regularizers.L2(0.001))(tf.concat([g[idx - 1], f[idx]], axis=-1))
            h[idx] = keras.layers.Conv2D(filters=outputs_size[idx], kernel_size=3, activation="relu", padding="same",
                                         kernel_regularizer=keras.regularizers.L2(0.001))(conv_1x1)

        # Once the "merged feature map" is defined, attach the "merge base" to it:
        if idx <= 2:
            g[idx] = keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last", interpolation="bilinear")(h[idx])
        else:
            g[idx] = keras.layers.Conv2D(filters=outputs_size[idx], kernel_size=3, activation="relu", padding="same",
                                          kernel_regularizer=keras.regularizers.L2(0.001))(h[idx])

    # Once the model is completed, define the output layer of the score, geometric, and angle maps:
    score_map = keras.layers.Conv2D(filters=1, kernel_size=1, activation=tf.nn.sigmoid, padding="same",
                                    kernel_regularizer=keras.regularizers.L2(0.001))(g[3])
    geo_map = (keras.layers.Conv2D(filters=4, kernel_size=1, activation=tf.nn.sigmoid, padding="same",
                                   kernel_regularizer=keras.regularizers.L2(0.001))(g[3])*512)
    angle_map = ((keras.layers.Conv2D(filters=1, kernel_size=1, activation=tf.nn.sigmoid, padding="same",
                                      kernel_regularizer=keras.regularizers.L2(0.001))(g[3]) - 0.5) * np.pi/2)

    # Merge the geometric map with the angle map and prepare the model output:
    complete_geo_map = tf.concat([geo_map, angle_map], axis=-1)
    output_ly = tf.concat([complete_geo_map, score_map], axis=-1)

    # Create the model and return it:
    model = keras.Model(inputs=input_ly, outputs=[output_ly], name="TFM_EAST")
    return model
####################################################################################################
####################################################################################################
####################################################################################################
print("\t- File \"TextDetection_Functions.py\" loaded successfully.\n")
