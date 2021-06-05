import numpy as np
import src.config as C
import random
import copy
import cv2
import pandas as pd
import os
# import PIL
from PIL import Image

# def get_img_output_length(width, height):
#     def get_output_length(input_length):
#         return input_length//16

#     return get_output_length(width), get_output_length(height)


def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)


def calc_rpn(img_data, width, height):
	"""(Important part!) Calculate the rpn for all anchors 
		If feature map has shape 38x50=1900, there are 1900x9=17100 potential anchors
	
	Args:
		C: config
		img_data: augmented image data
		width: original image width (e.g. 600)
		height: original image height (e.g. 800)

	Returns:
		y_rpn_cls: list(num_bboxes, y_is_box_valid + y_rpn_overlap)
			y_is_box_valid: 0 or 1 (0 means the box is invalid, 1 means the box is valid)
			y_rpn_overlap: 0 or 1 (0 means the box is not an object, 1 means the box is an object)
			final shape: (None, num_anchors*anchors_ratio*2, final_net_layer_x, final_net_layer_y)
		y_rpn_regr: list(num_bboxes, 4*y_rpn_overlap + y_rpn_regr)
			y_rpn_regr: x1,y1,x2,y2 bounding boxes coordinates
			final shape: (None, num_anchors*anchors_ratio*2*4, final_net_layer_x, final_net_layer_y)

	"""
	downscale = float(C.rpn_stride)
	anchor_sizes = C.anchor_box_scales   # 128, 256, 512
	anchor_ratios = C.anchor_box_ratios  # 1:1, 1:2*sqrt(2), 2*sqrt(2):1
	num_anchors = len(anchor_sizes) * len(anchor_ratios) # 3x3=9

	# calculate the output map size based on the network architecture
	(output_width, output_height) = (width//C.in_out_img_size_ratio, height//C.in_out_img_size_ratio)
	n_anchratios = len(anchor_ratios)    # 3
	
	# initialise empty output objectives
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

	# num_bboxes = len(img_data['bboxes'])
	num_bboxes = img_data.shape[0]

	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
	best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

	# get the GT box coordinates, and resize to account for image resizing
	gta = np.zeros((num_bboxes, 4))
	# for bbox_num, bbox in enumerate(img_data['bboxes']): # previously
	for bbox_num, bbox in img_data.iterrows():
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = bbox.x1s
		gta[bbox_num, 1] = bbox.x2s
		gta[bbox_num, 2] = bbox.y1s
		gta[bbox_num, 3] = bbox.y2s
	
	# rpn ground truth

	for anchor_size_idx in range(len(anchor_sizes)):
		for anchor_ratio_idx in range(n_anchratios):
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
			
			for ix in range(output_width):					                                # xi è un pixel nell'immagine 37x37
				# x-coordinates of the current anchor box	
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2                              # Ad ogni pixel dell'immagine 37x37 corrispondono 16px nell'immagine originale
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
				
				# ignore boxes that go across image boundaries					
				if x1_anc < 0 or x2_anc > width:
					continue
					
				for jy in range(output_height):

					# y-coordinates of the current anchor box
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# ignore boxes that go across image boundaries
					if y1_anc < 0 or y2_anc > height:
						continue

					# bbox_type indicates whether an anchor should be a target
					# Initialize with 'negative'
					bbox_type = 'neg'

					# this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
					best_iou_for_loc = 0.0

					for bbox_num in range(num_bboxes):
						
						# get IOU of the current GT box and the current anchor box
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
						# calculate the regression targets if they will be needed
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap: #Al primo giro best_iou_for_bbox[bbox_num] è 0, poi viene aggiornato e verrà ricontrollato al prossimo giro del for padre
							cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0        # Qui stiamo lavorando nelle dimensioni originali (es. 600x600)
							cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
							cxa = (x1_anc + x2_anc)/2.0
							cya = (y1_anc + y2_anc)/2.0

							tx = (cx - cxa) / (x2_anc - x1_anc)         # Shift tra i centri di gt e anchor
							ty = (cy - cya) / (y2_anc - y1_anc)
							tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
							th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
						
						if img_data.loc[bbox_num]['CLASS'] != 'bg':

							# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
							if curr_iou > best_iou_for_bbox[bbox_num]:
								best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
								best_iou_for_bbox[bbox_num] = curr_iou
								best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
								best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

							# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
							if curr_iou > C.rpn_max_overlap:
								bbox_type = 'pos'
								num_anchors_for_bbox[bbox_num] += 1
								# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
								if curr_iou > best_iou_for_loc:
									best_iou_for_loc = curr_iou
									best_regr = (tx, ty, tw, th) #Regression layer target (y_true)

							# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
							if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
								# gray zone between neg and pos
								if bbox_type != 'pos':
									bbox_type = 'neutral'

					# turn on or off outputs depending on IOUs
                    # Le anchor valide sono quelle da "guardare", poi se c'è overlap è un True altrimenti False
					if bbox_type == 'neg':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'neutral':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'pos':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
						y_rpn_regr[jy, ix, start:start+4] = best_regr

	# we ensure that every bbox has at least one positive RPN region
	for idx in range(num_anchors_for_bbox.shape[0]):
        # Qui entra quando per una bbox non ha trovato ancore che superano 0.7
		if num_anchors_for_bbox[idx] == 0:
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1)) #TODO: check se effettivamente ha senso per noi
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

	num_pos = len(pos_locs[0])

	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.
	num_regions = 256

	if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr), num_pos


def augment(img_path, img_data_path, augment=True):
	# assert 'filepath' in img_data_path
	# assert 'bboxes' in img_data_path
	# assert 'width' in img_data
	# assert 'height' in img_data

	#path = os.path.join(config.TRAIN_PATCHES_FOLDER, '0_1638016380_205/')
	img_orig = np.load(img_path)
	imgData_orig = pd.read_pickle(img_data_path)

	img_aug = copy.copy(img_orig)
	imgData_aug = copy.deepcopy(imgData_orig)

	# resize paches, the final dimension is the one set in the config (C.resizeFinalDim)
	if C.resizePatch :
		rows, cols = img_orig.shape[:2]
		width_percent = (C.resizeFinalDim / float(cols))
		hsize = int((float(rows) * float(width_percent)))
		#img_orig = img_orig.resize((C.resizeFinalDim , hsize), PIL.Image.ANTIALIAS)

		for index, row in imgData_aug.iterrows():			
			x1 = row['x1s'] * width_percent
			x2 = row['x2s'] * width_percent
			y1 = row['y1s'] * width_percent
			y2 = row['y2s'] * width_percent
			imgData_aug.at[index,'x1s']= x1
			imgData_aug.at[index,'x2s']= x2
			imgData_aug.at[index,'y1s']= y1
			imgData_aug.at[index,'y2s']= y2

	if augment:
		rows, cols = img_orig.shape[:2]
		# print(cols)

		if C.use_horizontal_flips and np.random.randint(0, 2) == 0:
			# print('Horizontal flipping')
			img_aug = cv2.flip(img_aug, 1)
			for index, row in imgData_aug.iterrows():			
				x1 = row['x1s']
				x2 = row['x2s']
				imgData_aug.at[index,'x2s']= cols - x1
				imgData_aug.at[index,'x1s']= cols - x2

		if C.use_vertical_flips and np.random.randint(0, 2) == 0:
			# print('Vertical flipping')
			img_aug = cv2.flip(img_aug, 0)
			for index, row in imgData_aug.iterrows():			
				y1 = row['y1s']
				y2 = row['y2s']
				imgData_aug.at[index,'y2s']= rows - y1
				imgData_aug.at[index,'y1s']= rows - y2

		if C.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			# print("angle = ", angle) 
			if angle == 270:
				img_aug = np.transpose(img_aug, (1,0))
				img_aug = cv2.flip(img_aug, 0)
			elif angle == 180:
				img_aug = cv2.flip(img_aug, -1)
			elif angle == 90:
				img_aug = np.transpose(img_aug, (1,0))
				img_aug = cv2.flip(img_aug, 1)
			elif angle == 0:
				pass

			# print(type(imgData_aug))
			# print(imgData_aug.head())

			#for bbox in imgData_aug['bboxes']:
			for index, row in imgData_aug.iterrows():
				#row = imgData_aug.iloc[_]
				x1 = row['x1s']
				x2 = row['x2s']
				y1 = row['y1s']
				y2 = row['y2s']
				if angle == 270:
					# print("angle rot 270")
					# row['x1s'] = y1
					# row['x2s'] = y2
					# row['y1s'] = cols - x2
					# row['y2s'] = cols - x1
					imgData_aug.at[index,'x1s']=  y1
					imgData_aug.at[index,'x2s']=  y2
					imgData_aug.at[index,'y1s']=  cols - x2
					imgData_aug.at[index,'y2s']=  cols - x1	
				elif angle == 180:
					# print("angle rot 180")
					# row['x2s'] = cols - x1
					# row['x1s'] = cols - x2
					# row['y2s'] = rows - y1
					# row['y1s'] = rows - y2
					imgData_aug.at[index,'x2s'] = cols - x1
					imgData_aug.at[index,'x1s'] = cols - x2
					imgData_aug.at[index,'y2s'] = rows - y1
					imgData_aug.at[index,'y1s'] = rows - y2
				elif angle == 90:
					# print("angle rot 90")
					# row['x1s'] = rows - y2
					# row['x2s'] = rows - y1
					# row['y1s'] = x1
					# row['y2s'] = x2
					imgData_aug.at[index,'x1s'] = rows - y2
					imgData_aug.at[index,'x2s'] = rows - y1
					imgData_aug.at[index,'y1s'] = x1
					imgData_aug.at[index,'y2s'] = x2       
				elif angle == 0:
					pass

	#AA commentate perchè le nostre patches sono quadrate
	#imgData_aug['width'] = img_aug.shape[1]
	#imgData_aug['height'] = img_aug.shape[0]
	return img_aug, imgData_aug


#def get_anchor_gt( all_img_data, img_length_calc_function, mode='train'):
def get_anchor_gt(patches_path, patch_list, mode='train'):
	""" Yield the ground-truth anchors as Y (labels)
		
	Args:
		patch_list: list(filepath, width, height, list(bboxes))
		C: config
		img_length_calc_function: function to calculate final layer's feature map (of base model) size according to input image size
		mode: 'train' or 'test'; 'train' mode need augmentation

	Returns:
		x_img: image data after resized and scaling (smallest size = 300px)
		Y: [y_rpn_cls, y_rpn_regr]
		img_data_aug: augmented image data (original image with augmentation)
		debug_img: show image for debug
		num_pos: show number of positive anchors for debug
	"""
	while True:

		for patch_id in patch_list:
			print(patch_id)
			try:

				# read in image, and optionally add augmentation
				image_path = os.path.join(patches_path, patch_id, f"{patch_id}.npy")
				image_data_path = os.path.join(patches_path, patch_id, f"{patch_id}.pkl")

				if mode == 'train':
					x_img, img_data_aug  = augment(image_path, image_data_path, augment=True)
				else:
					x_img, img_data_aug  = augment(image_path, image_data_path, augment=False)

				(width, height) = x_img.shape
				debug_img = x_img.copy()

				try:
					y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(img_data_aug, width, height) # es.: (1, 60, 12, 12), (1, 240, 12, 12), 64

				except Exception as e:
					print(e)
					continue

				# Zero-center by mean pixel, and preprocess image
				zero_centering(x_img)

				# x_img[:, :, 0] -= C.img_channel_mean[0]
				# x_img[:, :, 1] -= C.img_channel_mean[1]
				# x_img[:, :, 2] -= C.img_channel_mean[2]
				# x_img /= C.img_scaling_factor

				x_img = np.expand_dims(x_img, axis=0) # (205, 205) --> (1, 205, 205)
				x_img = np.expand_dims(x_img, axis=3) # (1, 205, 205) --> (1, 205, 205, 1)

				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

				y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1)) # (1, 60, 12, 12) --> (1, 12, 12, 60)
				y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1)) # (1, 240, 12, 12) --> (1, 12, 12, 240)


				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug, debug_img, num_pos

			except Exception as e:
				print(e)
				continue


def zero_centering(img_patch):
    #in order to manage images with 3 channels. ours should be 1
    imageChannels = 1 if len(img_patch.shape)== 2 else 3
    
    if(imageChannels == 1):
        img_patch[:, :] -= C.img_channel_mean[0]
    else:
        img_patch[:, :, 0] -= C.img_channel_mean[0]
        img_patch[:, :, 1] -= C.img_channel_mean[1]
        img_patch[:, :, 2] -= C.img_channel_mean[2]
    
    return