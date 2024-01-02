print("\nATTENTION --> Loading functions from \"Text_Detection_and_Recognition_Functions.py\":")
####################################################################################################
###################################### LIBRARIES ###################################################
####################################################################################################
print("\t- Loading image and data management libraries: \"cv2\", \"numpy\", and \"shapely\".")
import cv2
import numpy as np
import shapely as shap
print("\t- Completed. Loading Deep Learning libraries \"tensorflow\" and \"keras\".")
import tensorflow as tf
from tensorflow import keras
print("\t- Completed. All libraries loaded.")
####################################################################################################
############################## MODEL 1 : models_assembler #########################################
####################################################################################################
# DESCRIPTION: Creates the necessary models for the text detection and recognition process. The input
# dictionary loads the requested models: EASTlite with "MobileNet" and the Chars74k classifier. It also
# includes the path with weights.
def models_assembler(models_dict):

    # Internal Function 1: Generates the EASTlite model
    def EASTlite():

        # Load the input layer along with the input preprocessing layer:
        input_ly = keras.Input(shape=[None, None, 3], dtype=tf.float32, name="Input_LY")
        input_ly = keras.applications.resnet.preprocess_input(input_ly)

        # Load the base model coupled with the input layer and initialize the list of layers:
        base_model = keras.applications.MobileNet(input_tensor=input_ly, include_top=False)
        base_model_layers = [] # Only the layers to be used will be stored here

        # Loop through all layers of the base model:
        for layer in base_model.layers:

            # Allow the layer to be trainable and store the layers that will be used:
            layer.trainable = True
            if layer.name in ["conv_pw_2_relu", "conv_pw_5_relu", "conv_pw_11_relu", "conv_pw_13_relu"]:
                base_model_layers.append(layer)

        # Initialize the streams of EASTlite outputs:
        base_model_layers.reverse() # From bottom to top
        f = [layer.output for layer in base_model_layers] # Feature maps
        h = [None, None, None, None] # Merged feature map
        g = [None, None, None, None] # Merge base
        outputs_size = [None, 128, 64, 32]

        # Loop through each state of both blocks of EASTlite:
        for idx in range(len(f)):

            # For index 0, both outputs coincide (f1 = h1):
            if idx == 0:
                h[idx] = f[idx]

            # For other indices, "h" is defined as Conv3x3(Conv1x1(...)):
            else:
                conv_1x1 = keras.layers.Conv2D(filters=outputs_size[idx], kernel_size=1, activation="relu", padding="same",
                                               kernel_regularizer=keras.regularizers.L2(0.001))(tf.concat([g[idx - 1], f[idx]], axis=-1))
                h[idx] = keras.layers.Conv2D(filters=outputs_size[idx], kernel_size=3, activation="relu", padding="same",
                                             kernel_regularizer=keras.regularizers.L2(0.001))(conv_1x1)

            # Once the "merged feature map" is defined, attach the "merge base" to it:
            if idx <= 2:
                g[idx] = keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last", interpolation="bilinear")(h[idx])
            else:
                g[idx] =  keras.layers.Conv2D(filters=outputs_size[idx], kernel_size=3, activation="relu", padding="same",
                                              kernel_regularizer=keras.regularizers.L2(0.001))(h[idx])

        # After completing the model, define the output layer for the score map, geometry, and angle:
        score_map = keras.layers.Conv2D(filters=1, kernel_size=1, activation=tf.nn.sigmoid, padding="same",
                                        kernel_regularizer=keras.regularizers.L2(0.001))(g[3])
        geo_map = (keras.layers.Conv2D(filters=4, kernel_size=1, activation=tf.nn.sigmoid, padding="same",
                                       kernel_regularizer=keras.regularizers.L2(0.001))(g[3])*512)
        angle_map = ((keras.layers.Conv2D(filters=1, kernel_size=1, activation=tf.nn.sigmoid, padding="same",
                                          kernel_regularizer=keras.regularizers.L2(0.001))(g[3]) - 0.5)* np.pi/2)

        # Combine the geometry map with the angle map and prepare the model output:
        complete_geo_map = tf.concat([geo_map, angle_map], axis=-1)
        output_ly = tf.concat([complete_geo_map, score_map], axis=-1)

        # Create the model and return it:
        model = keras.Model(inputs=input_ly, outputs=[output_ly], name="TFM_EAST")
        return model

    # Internal Function 2: Generates the Chars74k classification model
    def Chars74k_classification():

        # Load the input layer and data augmentation layers [1/8 -> pi/4] (slightly less rotation is used):
        input_ly = keras.Input(shape=[64, 64, 1], dtype=tf.float32)
        aug_ly = keras.layers.RandomRotation(factor=(-1/10, 1/10), fill_mode="nearest",
                                             interpolation="nearest", seed=177)(input_ly)
        aug_ly = keras.layers.RandomZoom(height_factor=(0, 0.2), width_factor=None, fill_mode="nearest",
                                         interpolation="nearest", seed=177)(aug_ly)
        aug_ly = keras.layers.RandomTranslation(height_factor=(0, 0.2), width_factor=(0, 0.2), fill_mode="nearest",
                                                interpolation="nearest", seed=177)(aug_ly)

        # Add the first convolutional layer:
        conv1_ly = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")(aug_ly)
        conv1_ly = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1_ly)

        # Add the second convolutional layer:
        conv2_ly = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu")(conv1_ly)
        conv2_ly = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2_ly)

        # Add the third convolutional layer:
        conv3_ly = keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu")(conv2_ly)
        conv3_ly = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3_ly)

        # Regularize from convolutional to dense and apply dropout:
        dropout_ly = keras.layers.Dropout(rate=0.5, seed=117)(conv3_ly)
        dense_ly = keras.layers.Flatten()(dropout_ly)

        # Add dense layers and the output of 62 classes:
        dense1_ly = keras.layers.Dense(units=150, activation="relu")(dense_ly)
        output_ly = keras.layers.Dense(units=62, activation="softmax")(dense1_ly)

        # Create the model and return it:
        model = keras.Model(inputs=input_ly, outputs=[output_ly], name="TFM_Charts74k")
        return model

    # Internal Function 3: Generates the EMNIST classification model
    def EMNIST_classification():

        # Load the input layer and data augmentation layers [1/8 -> pi/4] (slightly less rotation is used):
        input_ly = keras.Input(shape=[28, 28, 1], dtype=tf.float32)
        aug_ly = keras.layers.RandomRotation(factor=(-1/20, 1/20), fill_mode="nearest",
                                             interpolation="nearest", seed=177)(input_ly)
        aug_ly = keras.layers.RandomZoom(height_factor=(0, 0.05), width_factor=None, fill_mode="nearest",
                                         interpolation="nearest", seed=177)(aug_ly)
        aug_ly = keras.layers.RandomTranslation(height_factor=(0, 0.05), width_factor=(0, 0.05), fill_mode="nearest",
                                                interpolation="nearest", seed=177)(aug_ly)

        # Add the first convolutional layer:
        conv1_ly = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")(aug_ly)
        conv1_ly = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1_ly)

        # Add the second convolutional layer:
        conv2_ly = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu")(conv1_ly)
        conv2_ly = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2_ly)

        # Regularize from convolutional to dense and apply dropout:
        dropout_ly = keras.layers.Dropout(rate=0.5, seed=117)(conv2_ly)
        dense_ly = keras.layers.Flatten()(dropout_ly)

        # Add dense layers and the output of 47 classes:
        dense1_ly = keras.layers.Dense(units=100, activation="relu")(dense_ly)
        output_ly = keras.layers.Dense(units=47, activation="softmax")(dense1_ly)

        # Create the model and return it:
        model = keras.Model(inputs=input_ly, outputs=[output_ly], name="TFM_EMNIST")
        return model

    # (START OF MAIN) Initialize the dictionary that contains the model and extra information:
    ready_models = {"TextDetection":None, "TextRecognition":{"model":None, "input_size":None, "class":None}}

    # Load the EASTlite model, add its weights, and store it:
    EASTlite_model = EASTlite()
    EASTlite_model.load_weights(models_dict["EASTlite"])
    ready_models["TextDetection"] = EASTlite_model

    # Load the classification models, add their weights, and store them:
    if models_dict["Chars74k"] != None:
        Chars74k_model = Chars74k_classification()
        Chars74k_model.load_weights(models_dict["Chars74k"])
        ready_models["TextRecognition"]["model"] = Chars74k_model
        ready_models["TextRecognition"]["input_size"] = (64, 64) # Width x Height
        ready_models["TextRecognition"]["class"] = np.char.mod("%c", np.concatenate([np.arange(48, 57 + 1),
                                                                     np.arange(65, 90 + 1),
                                                                     np.arange(97, 122 + 1)])) #[0-9, A-Z, a-z]
    else:
        EMNIST_model = EMNIST_classification()
        EMNIST_model.load_weights(models_dict["EMNIST"])
        ready_models["TextRecognition"]["model"] = EMNIST_model
        ready_models["TextRecognition"]["input_size"] = (28, 28) # Width x Height
        ready_models["TextRecognition"]["class"] = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I",
                                                             "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b",
                                                             "d", "e", "f", "g", "h", "n", "q", "r", "t"])
        
    # Return the models ready for prediction:
    return ready_models
####################################################################################################
############################# FUNCTION 1: Text_Detection #########################################
####################################################################################################
# DESCRIPTION: This function receives a single image previously loaded in RGB format. It rescales
# the image to dimensions that are multiples of 32, allowing it to be fed into the EASTlite model.
# Subsequently, a prediction is made, and "bounding boxes" are regenerated from the predicted maps.
# The function's output is a vector of dimensions (X, 4, 2), where X is the number of distinct
# annotations, and 4 and 2 represent pairs of components (x, y) defining the vertices:
# ["x1","y1"], ["x2","y2"], ["x3","y3"], ["x4","y4"]). It also rearranges these "bounding boxes"
# into rows and columns, assuming each "bounding box" represents a detected word.
def Text_Detection(TD_model, img):

    # Internal Function 1: Obtains the vertex vector from the score and geometric maps
    def restore_rectangle(orig, geomt):

        # Extracts "Top-Right-Bottom-Left" and "angle" from the geometric map:
        TRDL = geomt[:, :4]
        angle = geomt[:, 4]

        # Adapts for the case of an angle greater than 0 (filtered):
        orig_0 = orig[angle >= 0]
        TRDL_0 = TRDL[angle >= 0]
        angle_0 = angle[angle >= 0]

        # If the angle is greater than 0, apply the following:
        if orig_0.shape[0] > 0:

            # Generates the matrix that condenses the geometric structure of the box:
            p = np.array([np.zeros(TRDL_0.shape[0]), -TRDL_0[:, 0] - TRDL_0[:, 2], TRDL_0[:, 1] + TRDL_0[:, 3],
                          -TRDL_0[:, 0] - TRDL_0[:, 2], TRDL_0[:, 1] + TRDL_0[:, 3], np.zeros(TRDL_0.shape[0]),
                          np.zeros(TRDL_0.shape[0]), np.zeros(TRDL_0.shape[0]), TRDL_0[:, 3], -TRDL_0[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # -> N*5*2

            # Determines the rotation matrices to be applied to the box in the x and y axes:
            rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # -> N*5*2
            rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            # Rotates the generated geometric matrix:
            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # -> N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # -> N*5*1
            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # -> N*5*2

            # Determines the 4 vertices of the polygon:
            p3_in_origin = orig_0 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # -> N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin
            new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # -> N*4*2
        else:
            new_p_0 = np.zeros((0, 4, 2))  # Case where the angle is negative

        # For the case of a negative angle, initialize the other point of the polygon:
        orig_1 = orig[angle < 0]
        TRDL_1 = TRDL[angle < 0]
        angle_1 = angle[angle < 0]

        # If the angle is less than 0, apply the following:
        if orig_1.shape[0] > 0:

            # Generates the matrix that condenses the geometric structure of the box (now negative angle):
            p = np.array([-TRDL_1[:, 1] - TRDL_1[:, 3], -TRDL_1[:, 0] - TRDL_1[:, 2], np.zeros(TRDL_1.shape[0]),
                          -TRDL_1[:, 0] - TRDL_1[:, 2], np.zeros(TRDL_1.shape[0]), np.zeros(TRDL_1.shape[0]),
                          -TRDL_1[:, 1] - TRDL_1[:, 3], np.zeros(TRDL_1.shape[0]), -TRDL_1[:, 1], -TRDL_1[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # -> N*5*2

            # Determines the rotation matrices to be applied to the box in the x and y axes (now negative angle):
            rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # -> N*5*2
            rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            # Rotates the generated geometric matrix (now negative angle):
            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # -> N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # -> N*5*1
            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # -> N*5*2

            # Determines the 4 vertices of the polygon (now negative angle):
            p3_in_origin = orig_1 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # -> N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin
            new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # -> N*4*2
        else:
            new_p_1 = np.zeros((0, 4, 2))  # Case where the angle is not negative

        # Returns the restoration of the text box:
        return np.concatenate([new_p_0, new_p_1])

    # Internal Function 2: Applies the "Non-Maximum Suppression" algorithm to the resulting boxes
    def NMS(raw_boxes, iou_threshold):

        # Sorts the box indices based on the score and initializes a list of selected indices:
        sorted_idx = np.argsort(raw_boxes[:, 8])[::-1]  # From highest to lowest "score"
        selected_idx = []  # Indices that will not be eliminated when applying the algorithm

        # Iterates over the sorted indices until no more indices are left:
        while len(sorted_idx) > 0:

            # Selects/adds the index with the highest score (the first one):
            actual_idx = sorted_idx[0]
            selected_idx.append(actual_idx)

            # Initializes the vectors that will store the IoU and generates the polygon of the selected box:
            IoU_array = np.zeros(len(raw_boxes))
            selected_poly = shap.geometry.Polygon(raw_boxes[actual_idx, :8].reshape((4, 2)))  # Excluding the score

            # Calculates the areas of the polygon/intersection between the selected and the rest:
            for idx in sorted_idx[1:]:

                # Generates the comparison polygon:
                other_poly = shap.geometry.Polygon(raw_boxes[idx, :8].reshape((4, 2)))  # Excluding the score

                # Determines if there is an intersection, and if so, calculates the IoU:
                if selected_poly.intersects(other_poly):
                    inter_area = selected_poly.intersection(other_poly).area
                    IoU_array[idx] = inter_area / (selected_poly.area + other_poly.area - inter_area)
                else:
                    IoU_array[idx] = 0.0

            # Determines which non-selected boxes meet the threshold and can be eliminated:
            overlap_idx = np.where(IoU_array >= iou_threshold)[0]
            sorted_idx = np.delete(sorted_idx, np.where(np.in1d(sorted_idx, overlap_idx))[0])

            # To continue iterating, remove the current stored index:
            sorted_idx = np.delete(sorted_idx, 0)

        # At the end, the list of selected indices is obtained, and the independent boxes are returned:
        return raw_boxes[selected_idx, :]

    # (START OF MAIN) Load dimensions of the incoming image:
    height, width, _ = img.shape

    # Calculate the side limitation ratio according to the maximum permissible:
    if max(height, width) > 2400:
        ratio = (2400 / height if height > width else 2400 / width)
    else:
        ratio = 1.0

    # Correct the sides so that they do not exceed the limitation ratio:
    height_rs = int(height * ratio)
    width_rs = int(width * ratio)

    # Limit these dimensions to multiples of 32 and generate the new dimensions:
    height_rs = height_rs if height_rs % 32 == 0 else (height_rs // 32 - 1) * 32
    width_rs = width_rs if width_rs % 32 == 0 else (width_rs // 32 - 1) * 32
    height_rs = max(32, height_rs)
    width_rs = max(32, width_rs)

    # Resize the image to the new dimensions, make the prediction, and segment the predicted maps:
    img_rs = cv2.resize(img.copy(), (int(width_rs), int(height_rs)))
    pred_maps = TD_model.predict(img_rs[np.newaxis, :, :, :], batch_size=1)
    geo_maps = np.array(pred_maps[0, :, :, :5])  # Geometric map "Top-Right-Bottom-Left-Angle"
    score_map = np.array(pred_maps[0, :, :, 5])  # Score map

    # Filtering the score map by "threshold" and sorting the rectangles by the y-axis:
    xy_text = np.argwhere(score_map > 0.8)
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    # Restore the rectangles from the maps and adjust to the appropriate format:
    box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_maps[xy_text[:, 0], xy_text[:, 1], :])  # -> N*4*2
    boxes = np.zeros((box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    # NOTE: The internal function "restore_rectangle" manages to generate the "bounding boxes" from
    # the geometric and score maps in a format (X, 4, 2), i.e., X boxes, 4 vertices of 2 coordinates
    # (x, y). Subsequently, the format must be changed to (X, 9), i.e., X boxes and 8 components
    # (4 vertices x 2 -> x1, y1, x2, y2, x3, y3, x4, y4) plus one more component for the score value.

    # Apply the NMS algorithm to group the detected boxes:
    boxes = NMS(boxes.astype("float32"), 0.2)

    # NOTE: Since the "lanms" package causes problems, an internal function known as
    # "NMS" will be created to replace it, although it is not the same algorithm. Its mission is to reduce
    # the number of recurring boxes and return the remaining boxes in format (X', 9). Later, another
    # filtering will be applied, and it will finally return to the format (X', 4, 2). Where X' > X.

    # Try to perform the filtering and sorting process on the "bounding boxes" that have survived NMS:
    try:

        # Filter out some low-score boxes using the average score map (different from the original EAST):
        for ii, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32)//4, 1)
            boxes[ii, 8] = cv2.mean(score_map, mask)[0]

        # Final filtering is done based on the box detection threshold:
        boxes = boxes[boxes[:, 8] > 0.1]
        
        # Adjust their final dimensions to a vector [x1, y1], ..., [x4, y4] and the resizing ratio:
        boxes = boxes[:, :8].reshape(-1, 4, 2)
        boxes[:, :, 0] /= width_rs / float(width)  # Scaling ratio in "width"
        boxes[:, :, 1] /= height_rs / float(height)  # Scaling ratio in "height"
        boxes = boxes.astype(np.int32)  #-> From here, the boxes are sorted !!!

        # Calculate the centroids (CDG) of each of the "bounding boxes" and rearrange the indices according to Y:
        cdgs = np.column_stack((np.sum(boxes[:, :, 0], axis=1) / boxes[:, :, 0].shape[1], np.sum(boxes[:, :, 1], axis=1) / boxes[:, :, 1].shape[1]))
        idx_cdgs = np.argsort(cdgs[:, 1])

        # Initialize the list that will store each of the detected rows and the current row vector:
        box_rows = []
        actual_row = np.array([idx_cdgs[0]])

        # Iterate through each of the indices to group the "bounding boxes" by rows:
        for idx in idx_cdgs[1:]:

            # Calculate the difference in Y of the aspiring "bounding box" CDG and the current row CDG:
            dif_y = np.abs(cdgs[idx, 1] - cdgs[actual_row[-1], 1])

            # Check if the Y difference is a new line or the same:
            if dif_y <= 0.05*height:
                actual_row = np.append(actual_row, [idx]) #Same line
            else:
                box_rows.append(actual_row)
                actual_row =  np.array([idx])

            # NOTE: If the Y difference of the CDGs is greater than 5% of the height, it is another line.

        # Add the last row to the list of rows at the end of the loop:
        box_rows.append(actual_row)

        # Iterate through each of the detected rows and rearrange the indices according to X:
        for idx, row in enumerate(box_rows):
            box_rows[idx] = boxes[row[np.argsort(cdgs[row, 0])], :, :] #Directly introduce the "bounding boxes"

        # Return the regenerated and sorted "bounding boxes":
        return box_rows
        
    except:
        
        # Something went wrong in the previous process, so an empty set is returned:
        return []
####################################################################################################
########################## FUNCTION 2: Text_Recognition_by_Characters ############################
####################################################################################################
# DESCRIPTION: Responsible for finding characters within the bounding boxes provided by EASTlite.
# Once all contours within an EASTlite bounding box are located, they must be filtered and regenerated,
# sorted from left to right, and the absolute coordinates of those characters (EASTlite format) saved.
# Then each character is passed through the recognition model, and the message is extracted step by step.
# All character bounding boxes and the uncorrected message are returned.
def Text_Recognition_by_Characters(TR_model, img, TD_boxes):
    
    # Internal Function 1: Removes unwanted pixels in character detection and reorders
    def character_filter(all_chars_boxes):
        
        # Calculate the area of all possible character bounding boxes and get the clipping area:
        boxes_area = (all_chars_boxes[:, 1, 0] - all_chars_boxes[:, 0, 0])*(all_chars_boxes[:, 3, 1] - all_chars_boxes[:, 0, 1])
        area_lim = 0.2 * np.mean(boxes_area)

        # Divide all bounding boxes based on the area cutoff (characters) or not (loose pixels):
        char_boxes = all_chars_boxes[np.where(boxes_area >= area_lim)[0], :, :]
        pix_boxes = all_chars_boxes[np.where(boxes_area < area_lim)[0], :, :]

        # Apply the algorithm if there are loose pixels; otherwise, proceed to sorting directly:
        if len(pix_boxes) != 0:
            
            # Iterate through each of the pixels and check if it is the "i" dot (in the case of pixels):
            for pix_BB in pix_boxes:

                # Calculate the centroid of the pixel bounding box and possible "i" dot:
                cdg_coord = [np.sum(pix_BB[:, 0]) / pix_BB[:, 0].shape[0], np.sum(pix_BB[:, 1]) / pix_BB[:, 1].shape[0]]

                # Iterate through all bounding boxes that pass the area cutoff:
                for idx, char_BB in enumerate(char_boxes):

                    # Determine if the possible "i" dot is in a suitable position:
                    if (cdg_coord[0] >= np.min(char_BB[:, 0]) and cdg_coord[0] <= np.max(char_BB[:, 0]) and cdg_coord[1] < np.min(char_BB[:, 1])):

                        # Recalculate the bounding box coordinates to add the "i" dot (rectangular format):
                        char_boxes[idx, 0, :] = [np.minimum(char_BB[:, 0], pix_BB[:, 0]).min(), np.minimum(char_BB[:, 1], pix_BB[:, 1]).min()]
                        char_boxes[idx, 1, :] = [np.maximum(char_BB[:, 0], pix_BB[:, 0]).max(), np.minimum(char_BB[:, 1], pix_BB[:, 1]).min()]
                        char_boxes[idx, 2, :] = [np.maximum(char_BB[:, 0], pix_BB[:, 0]).max(), np.maximum(char_BB[:, 1], pix_BB[:, 1]).max()]
                        char_boxes[idx, 3, :] = [np.minimum(char_BB[:, 0], pix_BB[:, 0]).min(), np.maximum(char_BB[:, 1], pix_BB[:, 1]).max()]
                        break  # No need to look for another candidate
        
        # Calculate the CDGs and reorder characters from left to right (with X):
        cdgs = np.column_stack((np.sum(char_boxes[:, :, 0], axis=1) / char_boxes[:, :, 0].shape[1],
                                np.sum(char_boxes[:, :, 1], axis=1) / char_boxes[:, :, 1].shape[1]))
        chars_boxes_filtered = char_boxes[np.argsort(cdgs[:, 0]), :, :]

        # Return the filtered and sorted bounding boxes:
        return chars_boxes_filtered

    # Internal Function 2: Extracts characters and classifies them with the inserted model
    def character_recognition(TR_model, worl_crop, chars_boxes):
        
        # Initialize the vector that will store all character crops:
        crops_tensor = np.zeros((chars_boxes.shape[0], TR_model["input_size"][0], TR_model["input_size"][1], 1), dtype=np.float32)

        # Iterate through each character bounding box to form the tensor:
        for idx, BB in enumerate(chars_boxes):
            
            # Get the dimensions of the rectangle covering the bounding box and crop it:
            xbb, ybb, wbb, hbb = cv2.boundingRect(BB.astype(np.int32))
            crop_BB = worl_crop[ybb:(ybb + hbb), xbb:(xbb + wbb)]
            
            # Calculate the ratio and new dimensions respecting the aspect ratio:
            ratio = float(max(TR_model["input_size"])) / max((wbb, hbb))
            new_size = tuple([int(wbb * ratio), int(hbb * ratio)])  # Width x Height
            
            # Check that the new dimensions are suitable and resize the image:
            if new_size[0] > TR_model["input_size"][0] or new_size[1] > TR_model["input_size"][1]:
                ratio = float(min(TR_model["input_size"])) / min((wbb, hbb))
                new_size = tuple([int(wbb * ratio), int(hbb * ratio)])  # Width x Height
            crop_BB_rs = cv2.resize(crop_BB, new_size)  # Rescaled respecting the aspect ratio
            
            # Calculate the padding to add to match the network's input dimensions:
            delta_w = TR_model["input_size"][0] - new_size[0] if TR_model["input_size"][0] > new_size[0] else 0  # Padding in width
            delta_h = TR_model["input_size"][1] - new_size[1] if TR_model["input_size"][1] > new_size[1] else 0  # Padding in height
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            crop_BB_rs = cv2.copyMakeBorder(crop_BB_rs, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            
            # Add an extra dimension, normalize [0-1], and save the copy:
            crops_tensor[idx, :, :, :] = (np.expand_dims(crop_BB_rs, axis=-1) / 255).copy()
            
        # Pass the tensor to the recognition model and find the most probable characters:
        recog_hotshot = TR_model["model"].predict(crops_tensor, batch_size=crops_tensor.shape[0])
        max_idx = np.argmax(recog_hotshot, axis=1)  # Maximum of each sample
            
        # Return the detected word by joining all the characters:
        return TR_model["class"][max_idx].tobytes().decode("utf-8")
    
    # (START OF MAIN) Initialize the list that will store all character bounding boxes (per row):
    line_boxes = []
    
    # Initialize the message to be generated by character recognition:
    msg = ""
    
    # Iterate through each of the bounding boxes in each row:
    for row_BBs in TD_boxes:
        
        # Initialize the list that will store all character bounding boxes (per column/word):
        words_boxes = []
        
        # Iterate through each bounding box in the designated row:
        for BB in row_BBs:
            
            # Get the coordinates/dimensions of the box containing the bounding box and crop it:
            crop_x, crop_y, w, h = cv2.boundingRect(BB)
            img_crop = img[crop_y:(crop_y + h), crop_x:(crop_x + w), :]
            
            # Skip to the next word if the crop is empty:
            if not np.any(img_crop):
                continue
            
            # Convert the crop to BW and apply the "threshold" technique:
            crop_gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
            crop_retval, crop_thres = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            
            # Calculate the histogram and regularize it to the most used intensity level:
            if np.argmax(cv2.calcHist([crop_thres], [0], None, [256], [0, 256])) >= (255 / 2):
                crop_thres = cv2.bitwise_not(crop_thres)
            crop_thres = cv2.filter2D(crop_thres, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))  # Sharpening
                
            # NOTE: It is desired that the characters to be detected have a structure similar to the training of the
            # "Chars74k" classifier, that is, character in white and background in black. This will be marked by the
            # histogram; if the darkest intensity range, less than 255/2, is the most used, it means that the background
            # is black, as desired. Otherwise, the levels need to be inverted.
            
            # Get all contours (possible characters) and initialize the storage vector:
            crop_contours, crop_hierarchy = cv2.findContours(crop_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            word_arrays = np.array([]).reshape((0, 4, 2))  # EASTlite format

            # NOTE: The bounding boxes of the characters will store all of a single word/column, and
            # will be stored in EASTlite format: [x1, y1] for the upper-left corner, [x2, y2] for
            # the upper-right corner, [x3, y3] for the lower-right corner, and [x4, y4] for the lower-left
            # corner. In a contour detected by "cv2.boundingRect," the coordinates (X, Y) of the upper-left
            # corner [x1, y1] are given in EASTlite mode.
            
            # Iterate through each contour (possible character) detected:
            for crop_char in crop_contours:

                # Get the dimensions of the rectangle covering the detected contour:
                x, y, w, h = cv2.boundingRect(crop_char)

                # Calculate and store the coordinates (X, Y) of the characters in EASTlite mode:
                word_arrays = np.append(word_arrays, [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])], axis=0)
                
            # Filter the detected characters and sort them from left to right:
            word_arrays = character_filter(word_arrays.copy())
            
            # Classify each of the detected characters to generate the examined word:
            msg = msg + character_recognition(TR_model, crop_thres, word_arrays)
            
            # Update local coordinates to absolute (use of the upper-left corner [x1, y1] in EASTlite):
            word_arrays[:, :, 0] += crop_x
            word_arrays[:, :, 1] += crop_y

            # Store the absolute bounding boxes in the list of words:
            words_boxes.append(word_arrays)  # Filtered and sorted
            msg = msg + " "  # Space between words

        # Each group of words is stored in the list of rows:
        line_boxes.append(words_boxes)
        msg = msg[:-1] + "\n"  # Space between lines
        
    # Return the filtered and sorted list of lists of character bounding boxes, along with the message:
    return line_boxes, msg[:-1].replace("\x00", "").lower()  # Remove the code and convert to lowercase
####################################################################################################
########################## FUNCTION 3: message_corrector ###########################################
####################################################################################################
# DESCRIPTION: It is responsible for correcting the message word by word, leaving it unchanged if the correction
# would result in exceeding the original word by 2 syllables. The message is returned in the same structure but corrected.
def message_corrector(msg, corrector):
    
    # Initialize the corrected message to be generated:
    msg_correct = ""
    
    # Iterate through each of the word rows:
    for line in msg.split("\n"):

        # Iterate through each word in each line:
        for word in line.split(" "):
            
            # Check if the word needs correction:
            if not corrector.check(word):
                
                # Get its alternatives, and if there are none, leave it as it is:
                sugest_word = corrector.suggest(word)
                if sugest_word:
                
                    # Get the number of syllables for each alternative and their difference from the real one:
                    sugest_chars_num = np.array([len(alt) for alt in sugest_word])
                    diff = np.abs(sugest_chars_num - len(word))
                    
                    # Choose the index of the suggestion with the smallest difference:
                    idx_min_dif = np.argmin(diff)
                    
                    # If the difference is greater than 2, leave it as it is, otherwise choose that one:
                    if diff[idx_min_dif] < 2:
                        word = sugest_word[idx_min_dif]

            # Add the corrected word to the rest of the message:
            msg_correct = msg_correct + word + " "
        
        # Add the corrected line to the rest of the message:
        msg_correct = msg_correct + "\n"
        
    # Return the corrected message:
    return msg_correct
####################################################################################################
####################################################################################################
####################################################################################################
print("\t- File \"Text_Detection_and_Recognition_Functions.py\" loaded successfully.\n")
