import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.wcs as pywcs
import requests
from matplotlib.patches import Rectangle
import src.preprocessing as prep
import copy
import src.config as config
import time
from sklearn.metrics import average_precision_score, precision_score, recall_score






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
        print(f"Downloading {file} data...")
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


def rpn_to_roi(rpn_layer, regr_layer, use_regr=True, max_boxes=300, overlap_thresh=0.9): 
    """Convert rpn layer to roi bboxes

    Args: (num_anchors = 9)
        rpn_layer: output layer for rpn classification 
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 18) if resized image is 400 width and 300
        regr_layer: output layer for rpn regression
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 72) if resized image is 400 width and 300
        config: config
        use_regr: Wether to use bboxes regression in rpn
        max_boxes: max bboxes number for non-max-suppression (NMS)
        overlap_thresh: If iou in NMS is larger than this threshold, drop the box

    Returns:
        result: boxes from non-max-suppression (shape=(2000, 4))
            boxes: coordinates for bboxes (on the feature map)
    """
    regr_layer = regr_layer / config.std_scaling

    anchor_sizes = config.anchor_box_scales   # (3 in here)
    anchor_ratios = config.anchor_box_ratios  # (3 in here)

    assert rpn_layer.shape[0] == 1

    (rows, cols) = rpn_layer.shape[1:3]

    anchor_index = 0
    # print(rpn_layer.shape)

    # print(f'rpn_layer={rpn_layer.shape}' )
    # print(f'regr_layer={regr_layer.shape}' )

    # A.shape = (4, feature_map.height, feature_map.width, num_anchors) 
    # Might be (4, 18, 25, 18) if resized image is 400 width and 300
    # A is the coordinates for 9 anchors for every point in the feature map 
    # => all 18x25x9=4050 anchors cooridnates
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3])) # (4, 12, 12, num_anchor)

    # print(regr_layer)


    for anchor_size in anchor_sizes:
        # print('anchor_size=',anchor_size)
        for anchor_ratio in anchor_ratios:      
            # print('anchor_ratio=',anchor_ratio)
            # anchor_x = (128 * 1) / 16 = 8  => width of current anchor
            # anchor_y = (128 * 2) / 16 = 16 => height of current anchor
            anchor_x = (anchor_size * anchor_ratio[0])/config.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1])/config.rpn_stride

            # anchor_index: 0~8 (9 anchors)
            # the Kth anchor of all position in the feature map (9th in total)
            regr = regr_layer[0, :, :, 4 * anchor_index:4 * anchor_index + 4] # shape => (12, 12, 4)
            # print('REGR')
            # print(regr)
            
        
            # print(f'regr={regr.shape}')
            regr = np.transpose(regr, (2, 0, 1)) # shape => (4, 12, 12)

            # Create 18x25 mesh grid
            # For every point in x, there are all the y points and vice versa
            # X.shape = (18, 25)
            # Y.shape = (18, 25)
            X, Y = np.meshgrid(np.arange(cols),np.arange(rows))

            # Compute anchor position and size for each feature map point
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
            A[2, :, :, anchor_index] = np.maximum(1, A[2, :, :, anchor_index]) #TODO: capire se tenere --> sembra non funzionare            
            A[3, :, :, anchor_index] = np.maximum(1, A[3, :, :, anchor_index])
            

            # Convert (x, y , w, h) to (x1, y1, x2, y2)
            # x1, y1 is top left coordinate
            # x2, y2 is bottom right coordinate
            A[2, :, :, anchor_index] += A[0, :, :, anchor_index]
            A[3, :, :, anchor_index] += A[1, :, :, anchor_index]

            # with warnings.catch_warnings():
            #     warnings.filterwarnings('error')
            #     try:
            #         A[2, :, :, anchor_index] += A[0, :, :, anchor_index]
            #         A[3, :, :, anchor_index] += A[1, :, :, anchor_index]
            #     except Warning as w:
            #         print(w)
            #         print(A[2, :, :, anchor_index])
            #         print(A[0, :, :, anchor_index])
            #         print(A[3, :, :, anchor_index])
            #         print(A[1, :, :, anchor_index])
            #         return

            # Avoid bboxes drawn outside the feature map by clipping
            A[0, :, :, anchor_index] = np.maximum(0, A[0, :, :, anchor_index])
            A[1, :, :, anchor_index] = np.maximum(0, A[1, :, :, anchor_index])
            A[2, :, :, anchor_index] = np.minimum(cols-1, A[2, :, :, anchor_index])
            A[3, :, :, anchor_index] = np.minimum(rows-1, A[3, :, :, anchor_index])

            anchor_index += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # shape=(4320, 4)
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))                   # shape=(4320,)

    # Clipping nan
    nan_idx = np.where(np.isnan(all_boxes).sum(axis=1)>0)
    all_boxes = np.delete(all_boxes, nan_idx, 0)
    all_probs = np.delete(all_probs, nan_idx, 0)
    ###

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # Find out the bboxes which is illegal and delete them from bboxes list
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

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
        ppp = False #TODO: remove

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

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('error')
        #     try:
        #         w1 = np.exp(tw.astype(np.float128)) * w
        #     except Warning as warning:
        #         print('il problema è la width')
        #         np.save('./DEBUG/tw.npy', tw)
        #         print(warning)
        #         ppp=True

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('error')
        #     try:
        #         h1 = np.exp(th.astype(np.float128)) * h
        #     except Warning as warning:
        #         print('Il problema è la height')
        #         np.save('./DEBUG/th.npy', th)
        #         print(warning)
        #         ppp=True


        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        if ppp:
            print('w1')
            print(w1)
            print('h1')
            print(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def print_img(img_folder, img_name, data_folder=None, show_data=False):
    img = f'{img_folder}/{img_name}/{img_name}.npy'
    img = np.load(img)
    if data_folder==None:
        img_data = f'{img_folder}/{img_name}/{img_name}.pkl'
    else:
        img_data = f'{data_folder}/{img_name}/{img_name}.pkl'
    
    # if(config.convert_to_RGB):
    #     plt.imshow(img, cmap='viridis', vmax=255, vmin=0)   
    # else:
    # #img must be a numpy.ndarray img
    #     normalized_data = img * (1.0 /img.max())

    #     # Create figure and axes
    #     fig, ax = plt.subplots()

    #     # Display the image
    #     ax.imshow(normalized_data, cmap='viridis', vmax=1, vmin=0)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img, cmap='viridis', vmax=255, vmin=0)


    if img_data is None:
        return
    else:
        img_data = pd.read_pickle(img_data)
        if show_data: 
            display(img_data.iloc[:,20:])
        for _, box in img_data.iterrows():
            #box = df_scaled.loc[df_scaled['ID']==box_index].squeeze()
            ax.add_patch(Rectangle((box['x1s'] , box['y1s']), box['x2s'] - box['x1s'], box['y2s'] - box['y1s'], linewidth=.5, edgecolor='r',facecolor='none'))
            #plt.text(box.x - patch_xo, box.y - patch_yo, box_index, fontsize = 1)
        
        plt.show()
        
    return

def calc_iou(R, img_data, class_mapping):
    """Converts from (x1,y1,x2,y2) to (x,y,w,h) format

    Args:
        R: bboxes, probs
    """
    
    gta = np.zeros((img_data.shape[0], 4))
    # print(R)
    # print(img_data)

    for box_index, bbox in img_data.iterrows():
        
        # get the GT box coordinates, and resize to account for image resizing
        # gta[box_index, 0] = (40 * (600 / 800)) / 16 = int(round(1.875)) = 2 (x in feature map)
        gta[box_index, 0] = int(round(bbox['x1s']/config.rpn_stride))
        gta[box_index, 1] = int(round(bbox['x2s']/config.rpn_stride))
        gta[box_index, 2] = int(round(bbox['y1s']/config.rpn_stride))
        gta[box_index, 3] = int(round(bbox['y2s']/config.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] # for debugging only

    # R.shape[0]: number of bboxes (=2000 from non_max_suppression)
    for ix in range(R.shape[0]):
        # print(f'ix ={ix}')
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

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
        if best_iou < config.classifier_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h]) # ROI proposta dal network
            IoUs.append(best_iou) # Quanto x_roi overlappa con la gt

            # bg=0
            if config.classifier_min_overlap <= best_iou < config.classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
                # bg+=1

            elif config.classifier_max_overlap <= best_iou:
                cls_name = str(img_data.loc[box_index,'class_label'])
                # print(best_bbox)
                # print('!',gta[best_bbox])
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                # print('cg:', cxg,cyg)

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                # print('c:', cx,cy)

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
                
                # print(x1, x2, y1, y2)
                # print(tx, ty, tw, th)

            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError
        
        # print('bg=',bg)
        # One-hot encodig array of class
        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0] #[0,0,0,..]
        class_label[class_num] = 1. #[0,1,0,..]
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0.] * 4 * (len(class_mapping)-1) # -1 for the 'bg' class
        labels = [0.] * 4 * (len(class_mapping)-1) # -1 for the 'bg' class
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = config.classifier_regr_std #TODO: da capire
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    # bboxes that iou > config.classifier_min_overlap for all gt bboxes in 2000 non_max_suppression bboxes
    X = np.array(x_roi)
    # one hot encode for bboxes from above => x_roi (X)
    Y1 = np.array(y_class_num)
    # corresponding labels and corresponding gt bboxes
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs

def plot_loss(history):
    
    r_epochs = history.shape[0]
    total_loss = np.sum(history[:,:4],axis=1)

    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), total_loss, 'r')
    plt.title('Total loss')
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), history[:,4], 'r')
    plt.title('class_acc')
    plt.show()

    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), history[:, 0], 'r')
    plt.title('loss_rpn_cls')
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), history[:, 1], 'r')
    plt.title('loss_rpn_regr')
    plt.show()


    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), history[:, 2], 'r')
    plt.title('loss_detector_cls')
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), history[:, 3], 'r')
    plt.title('loss_detector_regr')
    plt.show()
    
    return

def get_real_coordinates(x1, y1, x2, y2):
    ratio = config.resizeFinalDim / config.patch_dim if config.resizePatch else 1.
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)

def get_detections(patch_id, bboxes, probs):
    boxes_coords ={'x1s':[], 'y1s':[], 'x2s':[], 'y2s':[], 'class':[], 'prob':[]}
    
    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2) #TODO: sposta soglia in config
       
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]
            print((x1, y1, x2, y2))
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(x1, y1, x2, y2)

            boxes_coords['x1s'].append(real_x1)
            boxes_coords['y1s'].append(real_y1)
            boxes_coords['x2s'].append(real_x2)
            boxes_coords['y2s'].append(real_y2)
            boxes_coords['class'].append(key)
            boxes_coords['prob'].append(new_probs[jk])
    
    if not os.path.exists(os.path.join(config.EVAL_RESULTS, f"{patch_id}")):
        os.makedirs(os.path.join(config.EVAL_RESULTS, f"{patch_id}/"))
        a = pd.DataFrame.from_dict(boxes_coords)        
        a.to_pickle(f'{config.EVAL_RESULTS}/{patch_id}/{patch_id}.pkl')

    return boxes_coords

def get_predictions(image, class_list, acceptance_treshold, rpn_model, detector_model):
    start = time.time()
    class_mapping = {key:value for key, value in enumerate(class_list)}
    class_mapping[len(class_mapping)] = 'bg'

    print(class_mapping)

    print('Predict')

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = rpn_model.predict_on_batch(image)
    # [Y1, Y2] = rpn_model.predict_on_batch(image)

    print('rpn_to_roi')

    # print(Y1.shape, Y2.shape)

    R = rpn_to_roi(Y1, Y2, overlap_thresh=0.9, max_boxes=64)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // config.num_rois + 1):
        ROIs = np.expand_dims(R[config.num_rois * jk:config.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // config.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], config.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        # print(ROIs)

        [P_cls, P_regr] = detector_model.predict([F, ROIs])
        # [P_cls, P_regr] = detector_model.predict([image, ROIs])
        print(P_cls)
        print(P_regr)
        for ii in range(P_cls.shape[1]):
            # if classification perc is too low OR it is. a 'bg' image THEN discard
            if np.max(P_cls[0,ii,:]) < acceptance_treshold or np.argmax(P_cls[0,ii,:]) == (P_cls.shape[2] - 1):
                continue

            cls_num = np.argmax(P_cls[0, ii, :])
            cls_name = class_mapping[cls_num]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []
            (x,y,w,h) = ROIs[0,ii,:]
            
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= config.classifier_regr_std[0]
                ty /= config.classifier_regr_std[1]
                tw /= config.classifier_regr_std[2]
                th /= config.classifier_regr_std[3]
                
                X, T = np.zeros((4, 1, 1)), np.zeros((4, 1, 1))
                
                X[0,0,0] = x
                X[1,0,0] = y
                X[2,0,0] = w
                X[3,0,0] = h
                
                T[0,0,0] = tx
                T[1,0,0] = ty
                T[2,0,0] = tw
                T[3,0,0] = th
                
                [[[x]], [[y]], [[w]], [[h]]] = apply_regr_np(X, T)

            except Exception as e:
                print('Exception: {}'.format(e))
                pass

            bboxes[cls_name].append([config.rpn_stride*x, config.rpn_stride*y, config.rpn_stride*(x+w), config.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))
   
    print(f'Elapsed:{time.time()-start}')
    return bboxes, probs

def evaluate_model(rpn_model, detector_model, backbone, val_patch_list, class_list, metric_threshold):

    preds = {}
    mAP = []
    mPrec = []
    mRecall = []
    val_datagen = prep.get_anchor_gt(config.TRAIN_PATCHES_FOLDER, val_patch_list, backbone=backbone, mode='eval', infinite_loop=False)

    for patch in val_datagen:
        image, _, _, _, _, patch_id = patch
        bboxes, probs = get_predictions(image, class_list, acceptance_treshold=.4, rpn_model=rpn_model, detector_model=detector_model)
        
        # print(bboxes, probs)
        
        detections = get_detections(patch_id, bboxes, probs)
        macro_AP, macro_prec, macro_recall = get_img_scores(detections, patch_id, metric_threshold)
        preds[patch_id] = {'bboxes':bboxes, 'probs':probs, 'mAP':macro_AP, 'macro_precision':macro_prec, 'macro_recall':macro_recall}
        mAP.append(macro_AP)
        mPrec.append(macro_prec)
        mRecall.append(macro_recall)


    total_mAP = np.array(mAP).mean()
    total_mPrec = np.array(mPrec).mean()
    total_mRecall = np.array(mRecall).mean()

    print(f'\nTotal model metrics: mAP: {total_mAP} - macro_precision: {total_mPrec} - macro_recall: {total_mRecall}')
    return preds, total_mAP, total_mPrec, total_mRecall

def compute_map(y_pred, gt_patch_id, data_folder):
    T = {}
    P = {}
    f = config.patch_dim / float(config.resizeFinalDim) if config.resizePatch else 1.

    gt = pd.read_pickle(f'{data_folder}/{gt_patch_id}/{gt_patch_id}.pkl')

    gt['bbox_matched'] = False
    pred_probs = np.array(y_pred['prob'])

    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    # print('708')
    # print(y_pred)

    for box_idx in box_idx_sorted_by_prob:
        # pred_box = y_pred.iloc[box_idx,:]
        pred_class = y_pred['class'][box_idx]
        pred_x1 = y_pred['x1s'][box_idx]
        pred_x2 = y_pred['x2s'][box_idx]
        pred_y1 = y_pred['y1s'][box_idx]
        pred_y2 = y_pred['y2s'][box_idx]
        pred_prob = y_pred['prob'][box_idx]
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for idx, gt_box in gt.iterrows():
            gt_class = gt_box['class_label']
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            gt_x1 = gt_box['x1s']/f
            gt_x2 = gt_box['x2s']/f
            gt_y1 = gt_box['y1s']/f
            gt_y2 = gt_box['y2s']/f
            iou = prep.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            if iou >= 0.5:
                found_match = True
                gt.at[idx,'bbox_matched'] = True
                break
            else:
                continue
        # display(gt)
        T[pred_class].append(int(found_match))

    for idx, gt_box in gt.iterrows():
        if not gt_box['bbox_matched']:
            if gt_box['class_label'] not in P:
                P[gt_box['class_label']] = []
                T[gt_box['class_label']] = []

            T[gt_box['class_label']].append(1)
            P[gt_box['class_label']].append(0)

    return T, P

def get_img_scores(detections, patch_id, metric_threshold):
    T = {}
    P = {}
    P_tresh = {}
    all_aps = []
    all_prec = []
    all_recall = []
    
    t, p = compute_map(detections, patch_id, config.TRAIN_PATCHES_FOLDER)
    # print('762')
    # print(t, p)

    for key in t.keys():
        if key not in T:
            T[key] = []
            P[key] = []
        T[key].extend(t[key])
        P[key].extend(p[key])


    for key in T.keys():
        ap = average_precision_score(T[key], P[key])
        # print('773')
        # print(T[key], P[key])
        P_tresh[key] = np.where(np.array(P[key]) > metric_threshold, 1, 0)

        prec = precision_score(T[key], P_tresh[key], zero_division=0)
        recall = recall_score(T[key], P_tresh[key], zero_division=0)
        # print('{} AP: {}'.format(key, ap))
        all_aps.append(ap)
        all_prec.append(prec)
        all_recall.append(recall)

    mAP=np.mean(np.array(all_aps))
    macro_prec = np.mean(np.array(all_prec))
    macro_recall =  np.mean(np.array(all_recall))

    # print('mAP = {}'.format(mAP))
    return mAP, macro_prec, macro_recall

def get_model_last_checkpoint(backbone):
    try:
    
        model_cp_dict = dict()
        model_dir = f'{config.MODEL_WEIGHTS}/{backbone}'
        print(f'Checking model checkpoints in directory {config.MODEL_WEIGHTS}/{backbone}')
        for file in os.listdir(model_dir):
            if file.endswith(f"{backbone}.h5"):
                print(os.path.join(model_dir, file))
                key = file.split('_')[0]
                model_cp_dict[file] = int(key)

        cp = max(model_cp_dict, key=model_cp_dict.get)
    except:
        cp = None
    
    return cp

def merge_dols(dol1, dol2):
    """
    Merges two dictionary of lists.
    Example:
        d = {'a': ['1','2'], 'b':['8']}
        dd = {'a': ['3','4'], 'c': ['9']}
        become -> {'a': ['1', '2', '3', '4'], 'b': ['8'], 'c': ['9']}
    """
    keys = set(dol1).union(dol2)
    no = []
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)

def merge_dois(dol1, dol2):
    """
    Merges two dictionary of lists.
    Example:
        d = {'a': ['1','2'], 'b':['8']}
        dd = {'a': ['3','4'], 'c': ['9']}
        become -> {'a': ['1', '2', '3', '4'], 'b': ['8'], 'c': ['9']}
    """
    keys = set(dol1).union(dol2)
    no = 0
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w