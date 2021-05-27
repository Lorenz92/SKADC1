import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.wcs as pywcs
import os
import requests
from matplotlib.patches import Rectangle
import src.preprocessing as prep
import copy
import src.config as C


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_data(file, url, DOWNLOAD_FOLDER):
    data_path = os.path.join(DOWNLOAD_FOLDER, file)

    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    if os.path.exists(DOWNLOAD_FOLDER):
        print(f"Downloading SKA {file} data...")
        with requests.Session() as current_session:
            response = current_session.get(url, stream=True)
        save_response_content(response, data_path)
        print("Download completed!")

# BB plot test
def _get_bbox_from_ellipse(phi, r1, r2, cx, cy, h, w):
    """
    https://stackoverflow.com/questions/87734/
    how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse
    angle in degrees
    r1, r2 in number of pixels (half major/minor)
    cx and cy is pixel coordinated
    """
    half_pi = np.pi / 2
    ux = r1 * np.cos(phi)
    uy = r1 * np.sin(phi)
    vx = r2 * np.cos(phi + half_pi)
    vy = r2 * np.sin(phi + half_pi)

    hw = np.sqrt(ux * ux + vx * vx)
    hh = np.sqrt(uy * uy + vy * vy)
    x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh
    return (x1, y1, x2, y2)


def _gen_single_bbox(fits_fn, ra, dec, major, minor, pa):
    """
    Form the bbox BEFORE converting wcs to the pixel coordinates
    major and mior are in arcsec
    """
    ra = float(ra)
    dec = float(dec)
    fits_fn_dict = dict()

    if (fits_fn not in fits_fn_dict):
        hdulist = pyfits.open(fits_fn)
        height, width = hdulist[0].data.shape[0:2]
        w = pywcs.WCS(hdulist[0].header).deepcopy()
        fits_fn_dict[fits_fn] = (w, height, width)
    else:
        w, height, width = fits_fn_dict[fits_fn]

    cx, cy = w.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
    #cx = np.ceil(cx)
    if (not for_png):
        cx += 1
    #cy = np.ceil(cy)
    cy += 1
    if (cx < 0 or cx > width):
        print('got it cx {0}, {1}'.format(cx, fits_fn))
        return []
    if (cy < 0 or cy > height):
        print('got it cy {0}'.format(cy))
        return []
    if (for_png):
        cy = height - cy
    majorp = major / 3600.0 / pixel_res_x / 2 #actually semi-major 
    minorp = minor / 3600.0 / pixel_res_x / 2
    paa = np.radians(pa)
    x1, y1, x2, y2 = _get_bbox_from_ellipse(paa, majorp, minorp, cx, cy, height, width)
    # return x1, y1, x2, y2, height, width
    origin_area = (y2 - y1) * (x2 - x1)

    # crop it around the border
    xp_min = max(x1, 0)
    yp_min = max(y1, 0)
    xp_max = min(x2, width - 1)
    if (xp_max <= xp_min):
        return []
    yp_max = min(y2, height - 1)
    if (yp_max <= yp_min):
        return []
    new_area = (yp_max - yp_min) * (xp_max - xp_min)

    if (origin_area / new_area > 4):
        print('cropped box is too small, discarding...')
        return []
    return (xp_min, yp_min, xp_max, yp_max, height, width, cx, cy)


def rpn_to_roi(rpn_layer, regr_layer, use_regr=True, max_boxes=300, overlap_thresh=0.9):
    """Convert rpn layer to roi bboxes

    Args: (num_anchors = 9)
        rpn_layer: output layer for rpn classification 
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 18) if resized image is 400 width and 300
        regr_layer: output layer for rpn regression
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 72) if resized image is 400 width and 300
        C: config
        use_regr: Wether to use bboxes regression in rpn
        max_boxes: max bboxes number for non-max-suppression (NMS)
        overlap_thresh: If iou in NMS is larger than this threshold, drop the box

    Returns:
        result: boxes from non-max-suppression (shape=(300, 4))
            boxes: coordinates for bboxes (on the feature map)
    """
    regr_layer = regr_layer / C.std_scaling

    anchor_sizes = C.anchor_box_scales   # (3 in here)
    anchor_ratios = C.anchor_box_ratios  # (3 in here)

    assert rpn_layer.shape[0] == 1

    (rows, cols) = rpn_layer.shape[1:3]

    anchor_index = 0

    print(f'rpn_layer={rpn_layer.shape}' )
    print(f'regr_layer={regr_layer.shape}' )

    # A.shape = (4, feature_map.height, feature_map.width, num_anchors) 
    # Might be (4, 18, 25, 18) if resized image is 400 width and 300
    # A is the coordinates for 9 anchors for every point in the feature map 
    # => all 18x25x9=4050 anchors cooridnates
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3])) # (4, 12, 12, num_anchor)


    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:      
            # anchor_x = (128 * 1) / 16 = 8  => width of current anchor
            # anchor_y = (128 * 2) / 16 = 16 => height of current anchor
            anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride

            # anchor_index: 0~8 (9 anchors)
            # the Kth anchor of all position in the feature map (9th in total)
            regr = regr_layer[0, :, :, 4 * anchor_index:4 * anchor_index + 4] # shape => (12, 12, 4)
            # print(f'regr={regr.shape}')
            regr = np.transpose(regr, (2, 0, 1)) # shape => (4, 12, 12)

            # Create 18x25 mesh grid
            # For every point in x, there are all the y points and vice versa
            # X.shape = (18, 25)
            # Y.shape = (18, 25)
            X, Y = np.meshgrid(np.arange(cols),np.arange(rows))

            # Calculate anchor position and size for each feature map point
            # Praticamente qui X e Y rappresentano una griglia in cui ogni punto corrisponde ad una coppia di coordinate, es X[0] = [0,1,2,...], Y[0]=[0,0,0,...]
            # con X - anchor_x/2 evita di cilcare su ogni elemento ij della griglia e genera le origini delle ancore in un colpo solo.
            # Il motivo per usare la meshgrid e risparmiari il doppio ciclo innestato
            A[0, :, :, anchor_index] = X - anchor_x/2 # Top left x coordinate
            A[1, :, :, anchor_index] = Y - anchor_y/2 # Top left y coordinate
            A[2, :, :, anchor_index] = anchor_x       # width of current anchor
            A[3, :, :, anchor_index] = anchor_y       # height of current anchor

            # In A la prima dimensione seleziona la "coordinata", l'ultima seleziona l'ancora (nel senso di un tipo di ancora dato da aspect ratio e size), 
            # dim 1 e dim 2 sono le x e y dell'ancora sulla patch 12x12

            # Apply regression to x, y, w and h if there is rpn regression layer
            if use_regr:
                A[:, :, :, anchor_index] = apply_regr_np(A[:, :, :, anchor_index], regr) # Ora A è l'ancora modificata

            # Avoid width and height exceeding 1 by clipping values
            A[2, :, :, anchor_index] = np.maximum(1, A[2, :, :, anchor_index]) #TODO: capire se tenere
            A[3, :, :, anchor_index] = np.maximum(1, A[3, :, :, anchor_index])

            # Convert (x, y , w, h) to (x1, y1, x2, y2)
            # x1, y1 is top left coordinate
            # x2, y2 is bottom right coordinate
            A[2, :, :, anchor_index] += A[0, :, :, anchor_index]
            A[3, :, :, anchor_index] += A[1, :, :, anchor_index]

            # Avoid bboxes drawn outside the feature map by clipping
            A[0, :, :, anchor_index] = np.maximum(0, A[0, :, :, anchor_index])
            A[1, :, :, anchor_index] = np.maximum(0, A[1, :, :, anchor_index])
            A[2, :, :, anchor_index] = np.minimum(cols-1, A[2, :, :, anchor_index])
            A[3, :, :, anchor_index] = np.minimum(rows-1, A[3, :, :, anchor_index])

            anchor_index += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # shape=(4320, 4)
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))                   # shape=(4320,)

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # Find out the bboxes which is illegal and delete them from bboxes list
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    # print(f'all_boxes={all_boxes.shape}')

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    # print(f'all_boxes={all_boxes.shape}')

    # Apply non_max_suppression
    # Only extract the bboxes. Don't need rpn probs in the later process
    result, _ = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)

    return result

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the largest prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list 
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # np.testing.assert_array_less(x1, x2)
    # np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes 
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection
        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs

def apply_regr_np(X, T):
    """
    Apply regression layer to all anchors in one feature map
    Here we slightly correct position and dimensions of generated anchors that have fixed dimensions and are generated in specific points

    Args:
        X: shape=(4, 12, 12) the current anchor type for all points in the feature map
        T: regression layer shape=(4, 12, 12)

    Returns:
        X: regressed position and size for current anchor
    """
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx # Correzione dello shift del centro dovuta alla variazione di dimensione dell'ancora
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def print_img(img, img_data = None):
    #img must be a numpy.ndarray img

    normalized_data = img * (1.0 /img.max())

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(normalized_data, cmap='viridis', vmax=1, vmin=0)

    if img_data is None:
        return
    else:
        for _, box in img_data.iterrows():
            #box = df_scaled.loc[df_scaled['ID']==box_index].squeeze()
            ax.add_patch(Rectangle((box['x1s'] , box['y1s']), box['x2s'] - box['x1s'], box['y2s'] - box['y1s'], linewidth=.1, edgecolor='r',facecolor='none'))
            #plt.text(box.x - patch_xo, box.y - patch_yo, box_index, fontsize = 1)
        
        plt.show()
        
    return

def calc_iou(R, img_data, class_mapping):
    """Converts from (x1,y1,x2,y2) to (x,y,w,h) format

    Args:
        R: bboxes, probs
    """
    
    gta = np.zeros((img_data.shape[0], 4))

    for box_index, bbox in img_data.iterrows():
        # get the GT box coordinates, and resize to account for image resizing
        # gta[box_index, 0] = (40 * (600 / 800)) / 16 = int(round(1.875)) = 2 (x in feature map)
        gta[box_index, 0] = bbox['x1s']/C.rpn_stride
        gta[box_index, 1] = bbox['x2s']/C.rpn_stride
        gta[box_index, 2] = bbox['y1s']/C.rpn_stride
        gta[box_index, 3] = bbox['y2s']/C.rpn_stride

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] # for debugging only

    # R.shape[0]: number of bboxes (=300 from non_max_suppression)
    for ix in range(R.shape[0]):
        # print(f'ix ={ix}')
        (x1, y1, x2, y2) = R[ix, :]
        # x1 = int(round(x1))
        # y1 = int(round(y1))
        # x2 = int(round(x2))
        # y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        # Iterate through all the ground-truth bboxes to calculate the iou
        for box_index in range(img_data.shape[0]):
            # print(f'box_index={box_index}')
            curr_iou = prep.iou([gta[box_index, 0], gta[box_index, 2], gta[box_index, 1], gta[box_index, 3]], [x1, y1, x2, y2])

            # Find out the corresponding ground-truth box_index with larget iou
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = box_index
        
                
        # Discard ROI if overlap is not sufficient
        if best_iou < C.classifier_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)


            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'

            elif C.classifier_max_overlap <= best_iou:
                cls_name = str(img_data.loc[box_index,'CLASS']) #TODO: change here with the combination of 3 SIZE x 3 CLASS and remove str()
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        # One-hot encodig array of class
        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0] #[0,0,0,..]
        class_label[class_num] = 1. #[0,1,0,..]
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0.] * 4 * (len(class_mapping)-1) # -1 for the 'bg' class
        labels = [0.] * 4 * (len(class_mapping)-1) # -1 for the 'bg' class
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    print(IoUs)
    # bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
    X = np.array(x_roi)
    # one hot encode for bboxes from above => x_roi (X)
    Y1 = np.array(y_class_num)
    # corresponding labels and corresponding gt bboxes
    print(np.array(y_class_regr_label).shape,np.array(y_class_regr_coords).shape)
    print(np.array(y_class_regr_label))
    print(np.array(y_class_regr_coords))
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)
    print(Y2.shape)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs