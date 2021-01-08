import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.color import rgb2grey, rgb2hsv

import cv2


def channel_percentile(single_chan_image, percentile):
    sq_image = np.squeeze(single_chan_image)
    assert len(sq_image.shape) < 3

    thres_value = np.percentile(sq_image.ravel(), percentile)

    return float(thres_value) / 255.0


def high_value_region_mask(hsv_image, v_thres=0.6):
    if hsv_image.dtype == np.int:
        idx = (hsv_image[:, :, 2].astype(np.float) / 255.0) < v_thres
    else:
        idx = (hsv_image[:, :, 2].astype(np.float)) < v_thres
    mask = np.ones_like(hsv_image[:, :, 2])
    mask[idx] = 0
    return mask


def high_saturation_region_mask(hsv_image, s_thres=0.6):
    if hsv_image.dtype == np.int:
        idx = (hsv_image[:, :, 1].astype(np.float) / 255.0) < s_thres
    else:
        idx = (hsv_image[:, :, 1].astype(np.float)) < s_thres
    mask = np.ones_like(hsv_image[:, :, 1])
    mask[idx] = 0
    return mask


def get_masked_hue_values(rgb_image):
    """
    Get the pixels in the RGB image that has high saturation (S) and value (V) in HSV chanels

    :param rgb_image: image (height, width, channel)
    :return: a 1-d array
    """

    hsv_test_image = rgb2hsv(rgb_image)
    s_thres_val = channel_percentile(hsv_test_image[:, :, 1], percentile=30)
    v_thres_val = channel_percentile(hsv_test_image[:, :, 2], percentile=70)
    val_mask = high_value_region_mask(hsv_test_image, v_thres=v_thres_val)
    sat_mask = high_saturation_region_mask(hsv_test_image, s_thres=s_thres_val)
    masked_hue_image = hsv_test_image[:, :, 0] * 180
    # Note that the following statement is not equivalent to
    # masked_hue_1d= (maksed_hue_image*np.logical_and(val_mask,sat_mask)).ravel()
    # Because zero in hue channel means red, we cannot just set unused pixels to zero.
    masked_hue_1d = masked_hue_image[np.logical_and(
        val_mask, sat_mask)].ravel()

    return masked_hue_1d


def convert_to_hue_angle(hue_array):
    """
    Convert the hue values from [0,179] to radian degrees [-pi, pi]

    :param hue_array: array-like, the hue values in degree [0,179]
    :return: the angles of hue values in radians [-pi, pi]
    """

    hue_cos = np.cos(hue_array * np.pi / 90)
    hue_sine = np.sin(hue_array * np.pi / 90)

    hue_angle = np.arctan2(hue_sine, hue_cos)

    return hue_angle


def get_rgy_color_mask(hue_value, from_01=False):
    """
    return a tuple of np.ndarray that sets the pixels with red, green and yellow matrices to be true

    :param hue_value:
    :param from_01: True if the hue values is scaled from 0-1 (scikit-image), otherwise is -pi to pi
    :return:
    """

    n_hue_value = hue_value

    red_index = np.logical_and(n_hue_value < (
        0.125 * np.pi), n_hue_value > (-0.125 * np.pi))

    green_index = np.logical_and(
        n_hue_value > (0.66 * np.pi), n_hue_value < np.pi)

    yellow_index = np.logical_and(n_hue_value > (
        0.25 * np.pi), n_hue_value < (5.0 / 12.0 * np.pi))

    return red_index, green_index, yellow_index


def classify_color_by_range(hue_value):
    """
    Determine the color (red, yellow or green) in a hue value array

    :param hue_value: hue_value is radians
    :return: the color index ['red', 'yellow', 'green', '_', 'unknown']
    """

    red_index, green_index, yellow_index = get_rgy_color_mask(hue_value)

    color_counts = np.array([np.sum(red_index) / len(hue_value),
                             np.sum(yellow_index) / len(hue_value),
                             np.sum(green_index) / len(hue_value)])

    color_text = ['red', 'yellow', 'green', '_', 'unknown']

    min_index = np.argmax(color_counts)

    return min_index, color_text[min_index]


def classify_color_cropped_image(rgb_image):
    """
    Full pipeline of classifying the traffic light color from the traffic light image

    :param rgb_image: the RGB image array (height,width, RGB channel)
    :return: the color index ['red', 'yellow', 'green', '_', 'unknown']
    """

    hue_1d_deg = get_masked_hue_values(rgb_image)

    if len(hue_1d_deg) == 0:
        return 4, 'unknown'

    hue_1d_rad = convert_to_hue_angle(hue_1d_deg)

    return classify_color_by_range(hue_1d_rad)


def boxes_iou(box1, box2):

    # Get the Width and Height of each bounding box
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]

    # Calculate the area of the each bounding box
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2

    # Find the vertical edges of the union of the two bounding boxes
    mx = min(box1[0] - width_box1/2.0, box2[0] - width_box2/2.0)
    Mx = max(box1[0] + width_box1/2.0, box2[0] + width_box2/2.0)

    # Calculate the width of the union of the two bounding boxes
    union_width = Mx - mx

    # Find the horizontal edges of the union of the two bounding boxes
    my = min(box1[1] - height_box1/2.0, box2[1] - height_box2/2.0)
    My = max(box1[1] + height_box1/2.0, box2[1] + height_box2/2.0)

    # Calculate the height of the union of the two bounding boxes
    union_height = My - my

    # Calculate the width and height of the area of intersection of the two bounding boxes
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height

    # If the the boxes don't overlap then their IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_width * intersection_height

    # Calculate the area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate the IOU
    iou = intersection_area/union_area

    return iou


def nms(boxes, iou_thresh):

    # If there are no bounding boxes do nothing
    if len(boxes) == 0:
        return boxes

    # Create a PyTorch Tensor to keep track of the detection confidence
    # of each predicted bounding box
    det_confs = torch.zeros(len(boxes))

    # Get the detection confidence of each predicted bounding box
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]

    # Sort the indices of the bounding boxes by detection confidence value in descending order.
    # We ignore the first returned element since we are only interested in the sorted indices
    _, sortIds = torch.sort(det_confs, descending=True)

    # Create an empty list to hold the best bounding boxes after
    # Non-Maximal Suppression (NMS) is performed
    best_boxes = []

    # Perform Non-Maximal Suppression
    for i in range(len(boxes)):

        # Get the bounding box with the highest detection confidence first
        box_i = boxes[sortIds[i]]

        # Check that the detection confidence is not zero
        if box_i[4] > 0:

            # Save the bounding box
            best_boxes.append(box_i)

            # Go through the rest of the bounding boxes in the list and calculate their IOU with
            # respect to the previous selected box_i.
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]

                # If the IOU of box_i and box_j is higher than the given IOU threshold set
                # box_j's detection confidence to zero.
                if boxes_iou(box_i, box_j) > iou_thresh:
                    box_j[4] = 0

    return best_boxes


def detect_objects(model, img, iou_thresh, nms_thresh):

    # Start the time. This is done to calculate how long the detection takes.
    start = time.time()

    # Set the model to evaluation mode.
    model.eval()

    # Convert the image from a NumPy ndarray to a PyTorch Tensor of the correct shape.
    # The image is transposed, then converted to a FloatTensor of dtype float32, then
    # Normalized to values between 0 and 1, and finally unsqueezed to have the correct
    # shape of 1 x 3 x 416 x 416
    img = torch.from_numpy(img.transpose(
        2, 0, 1)).float().div(255.0).unsqueeze(0)

    # Feed the image to the neural network with the corresponding NMS threshold.
    # The first step in NMS is to remove all bounding boxes that have a very low
    # probability of detection. All predicted bounding boxes with a value less than
    # the given NMS threshold will be removed.
    list_boxes = model(img, nms_thresh)

    # Make a new list with all the bounding boxes returned by the neural network
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]

    # Perform the second step of NMS on the bounding boxes returned by the neural network.
    # In this step, we only keep the best bounding boxes by eliminating all the bounding boxes
    # whose IOU value is higher than the given IOU threshold
    boxes = nms(boxes, iou_thresh)

    # Stop the time.
    finish = time.time()

    # Print the time it took to detect objects
    print('\n\nIt took {:.3f}'.format(finish - start),
          'seconds to detect the objects in the image.\n')

    # Print the number of objects detected
    print('Number of Objects Detected:', len(boxes), '\n')

    return boxes


def load_class_names(namesfile):

    # Create an empty list to hold the object classes
    class_names = []

    # Open the file containing the COCO object classes in read-only mode
    with open(namesfile, 'r') as fp:

        # The coco.names file contains only one object class per line.
        # Read the file line by line and save all the lines in a list.
        lines = fp.readlines()

    # Get the object class names
    for line in lines:

        # Make a copy of each line with any trailing whitespace removed
        line = line.rstrip()

        # Save the object class name into class_names
        class_names.append(line)

    return class_names


def print_objects(boxes, class_names):
    print('Objects Found and Confidence Level:\n')
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%i. %s: %f' % (i + 1, class_names[cls_id], cls_conf))


def reutrn_color(textcolor):

    if textcolor == 'green':
        color = (0.0, 1.0, 0.0)
    else:
        color =(1.0, 0.0, 0.0)

    return color


def plot_boxes(img, boxes, class_names, plot_labels, color=None):

        # Define a tensor used to set the colors of the bounding boxes
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [
                               0, 1, 0], [1, 1, 0], [1, 0, 0]])

    # Define a function to set the colors of the bounding boxes
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))

        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]

        return int(r * 255)

    # Get the width and height of the image
    width = img.shape[1]
    height = img.shape[0]
    image_np = np.asarray(img)

    # Create a figure and plot the image
    fig, a = plt.subplots(1, 1)

    a.imshow(img)
    #img.save("test12.png", 'png')

    # Plot the bounding boxes and corresponding labels on top of the image
    for i in range(len(boxes)):

        # Get the ith bounding box
        box = boxes[i]
        cls_conf = box[5]
        cls_id = box[6]
        if class_names[cls_id] == 'traffic light':
            x1=0
            x2=0
            y1=0
            y2=0

            # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
            # of the bounding box relative to the size of the image.
            x1 = int(np.around((box[0] - box[2]/2.0) * width))
            y1 = int(np.around((box[1] - box[3]/2.0) * height))
            x2 = int(np.around((box[0] + box[2]/2.0) * width))
            y2 = int(np.around((box[1] + box[3]/2.0) * height))
            
            left, top, right, bottom = x1, y1, x2, y2
            #left, top, right, bottom = int(left * width_ratio), int(top * width_ratio), int(right * width_ratio), int(bottom * height_ratio)
           
            cropped_image = image_np[int(top):int(
                bottom), int(left):int(right), :]
            #hsv_test_image = rgb2hsv(cropped_image)
            detect_colors = classify_color_cropped_image(cropped_image)
            # Set the default rgb value to red
            couleur = reutrn_color(detect_colors[1])
            #import cv2
            print(couleur)

            #rgb = (1, 0, 0)
            #a.imshow(cropped_image)
            # cv2_imshow(cropped_image)
            cv2.rectangle(img, (left, top), (right, bottom), couleur, 2)
            #cv2.rectangle(img, (left, top), (right, bottom), couleur, 2)
            # Use the same color to plot the bounding boxes of the same object class
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes) / 255
                green = get_color(1, offset, classes) / 255
                blue = get_color(0, offset, classes) / 255

                # If a color is given then set rgb to the given color instead
                if color is None:
                    rgb = (red, green, blue)
                else:
                    rgb = color

            # Calculate the width and height of the bounding box relative to the size of the image.
            width_x = x2 - x1
            width_y = y1 - y2

            # Set the postion and size of the bounding box. (x1, y2) is the pixel coordinate of the
            # lower-left corner of the bounding box relative to the size of the image.
            rect = patches.Rectangle((x1, y2),
                                     width_x, width_y,
                                     linewidth=2,
                                     edgecolor=rgb,
                                     facecolor='none')

            # Draw the bounding box on top of the image
            a.add_patch(rect)

            # If plot_labels = True then plot the corresponding label
            if plot_labels:

                # Create a string with the object class name and the corresponding object class probability
                conf_tx = class_names[cls_id]+ ': {}'.format(detect_colors[1])

                # Define x and y offsets for the labels
                lxc = (img.shape[1] * 0.266) / 100
                lyc = (img.shape[0] * 1.180) / 100
                print("lefttttttttttttttttttttttttttt", rgb)
                # Draw the labels on top of the image
                a.text(x1 + lxc, y1 - lyc, conf_tx, fontsize=10, color=couleur,
                    

                       bbox=dict(facecolor='none', edgecolor=couleur, alpha=0.8))
    fig.set_size_inches(24.0, 14.0)
    # fig.savefig('temp.png')
    plt.rcParams['figure.figsize'] = [24.0, 14.0]
    plt.show()


def with_height(img, boxes, class_names, plot_labels, color=None):
    params = []
    # Define a tensor used to set the colors of the bounding boxes
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [
                               0, 1, 0], [1, 1, 0], [1, 0, 0]])

    # Define a function to set the colors of the bounding boxes
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))

        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]

        return int(r * 255)

    # Get the width and height of the image
    width = img.shape[1]
    height = img.shape[0]

    # Create a figure and plot the image

    # Plot the bounding boxes and corresponding labels on top of the image
    for i in range(len(boxes)):

        # Get the ith bounding box
        box = boxes[i]
        cls_conf = box[5]
        cls_id = box[6]
        if class_names[cls_id] == 'traffic light':

            # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
            # of the bounding box relative to the size of the image.
            x1 = int(np.around((box[0] - box[2]/2.0) * width))
            y1 = int(np.around((box[1] - box[3]/2.0) * height))
            x2 = int(np.around((box[0] + box[2]/2.0) * width))
            y2 = int(np.around((box[1] + box[3]/2.0) * height))

            params.append((x1, x2, y1, y2))

            #print("x1", x1, "x2", x2, "y2", y2)
    return params
