import numpy as np
from tensorflow.python.keras import backend
import src.config as C
import random
import copy
import cv2
import pandas as pd
import os
import PIL
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def calc_rpn(img_data, width, height, backbone):
	"""(Important part!) Calculate the rpn for all anchors 
		If feature map has shape 38x50=1900, there are 1900x9=17100 potential anchors
	
	Args:
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

	# print('71 - width:', width)
	# print('72 - height:', height)
	# display(img_data.iloc[: , 20:])

	downscale = float(C.rpn_stride)
	anchor_sizes = C.anchor_box_scales   # [32, 64, 128, 256, 512]
	anchor_ratios = C.anchor_box_ratios  # 1:1, 1:2, 2:1
	num_anchors = len(anchor_sizes) * len(anchor_ratios) # 3x5=15
	
	# calculate the output map size based on the network architecture
	if backbone == 'baseline_8':
		(output_width, output_height) = (width//2, height//2)
	elif backbone == 'baseline_44':
		(output_width, output_height) = (width//8, height//8)
	elif backbone == 'baseline_16':
		(output_width, output_height) = (width//4, height//4)
	elif backbone == 'vgg16':
		(output_width, output_height) = (width//C.in_out_img_size_ratio, height//C.in_out_img_size_ratio)
	elif backbone=='resnet50':
		(output_width, output_height) = (int(np.ceil(width/C.in_out_img_size_ratio)), int(np.ceil(height/C.in_out_img_size_ratio)))
	

	
	n_anchratios = len(anchor_ratios)    # 3
	
	# initialise empty output objectives
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))#.astype(int)
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors))#.astype(int)
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

	num_bboxes = img_data.shape[0]

	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
	best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int) #TODO: commentare astype
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32) #TODO: commentare astype

	# get the GT box coordinates, and resize to account for image resizing
	gta = np.zeros((num_bboxes, 4))
	for bbox_num, bbox in img_data.iterrows():
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = bbox.x1s
		gta[bbox_num, 1] = bbox.x2s
		gta[bbox_num, 2] = bbox.y1s
		gta[bbox_num, 3] = bbox.y2s
	
	# rpn ground truth

	for anchor_size_idx in tqdm(range(len(anchor_sizes))):
		for anchor_ratio_idx in range(n_anchratios):
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
			
			for ix in range(output_width):					                                # xi è un pixel nell'immagine 37x37
				# x-coordinates of the current anchor box	
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2                              # Ad ogni pixel dell'immagine 37x37 corrispondono 16px nell'immagine originale
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
				
				# ignore boxes that go across image boundaries					
				if (x1_anc < 0 or x2_anc >= width):
					continue
					
				for jy in range(output_height):

					# y-coordinates of the current anchor box
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# ignore boxes that go across image boundaries
					if (y1_anc < 0 or y2_anc >= height):
						continue

					# bbox_type indicates whether an anchor should be a target
					# Initialize with 'negative'
					bbox_type = 'neg'

					# this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
					best_iou_for_loc = 0.0

					for bbox_num in range(num_bboxes):

						# print('box_num', bbox_num)
						# print('anchor_x', anchor_x)
						# print('anchor_y', anchor_y)
						
						# get IOU of the current GT box and the current anchor box
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
						# print('anchor coords:', [x1_anc, y1_anc, x2_anc, y2_anc])
						# print('iou:',curr_iou)
						# calculate the regression targets if they will be needed
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap: #Al primo giro best_iou_for_bbox[bbox_num] è 0, poi viene aggiornato e verrà ricontrollato al prossimo giro del for padre
							cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0        # Qui stiamo lavorando nelle dimensioni originali (es. 600x600)
							cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
							cxa = (x1_anc + x2_anc)/2.0
							cya = (y1_anc + y2_anc)/2.0

							tx = (cx - cxa) / (x2_anc - x1_anc)         # Shift tra i centri di gt e anchor
							ty = (cy - cya) / (y2_anc - y1_anc)
							# tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
							# th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
							tw = 1.*(gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc)
							th = 1.*(gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc)
						## debug
						# if(ix==17 and jy==17):
						# 	print('17', (tx, ty, tw, th))
						# 	print(curr_iou)
						# 	print(anchor_ratio_idx + n_anchratios * anchor_size_idx)
							
						if img_data.loc[bbox_num]['class_label'] != 'bg':

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

							# if the IOU is >0.3 and <0.7, it is ambiguous and not included in the objective
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
						# y_rpn_regr[jy, ix, start:start+4] = best_regr
						y_rpn_regr[jy, ix, start:start+2 ] = best_regr[0:2]
						y_rpn_regr[jy, ix, start+2:start+4 ] = np.log(best_regr[2:])

	# we ensure that every bbox has at least one positive RPN region
	for idx in tqdm(range(num_anchors_for_bbox.shape[0])):
        # Qui entra quando per una bbox non ha trovato ancore che superano 0.7
		if num_anchors_for_bbox[idx] == 0:
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios * # y,x , anchor index
				best_anchor_for_bbox[idx,3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
			# y_rpn_regr[
			# 	best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+2] = best_dx_for_bbox[idx, 0:2]
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start+2:start+4] = np.log(best_dx_for_bbox[idx, 2:4])


	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1)) #ritorna 3 array: il primo sono gli indici di ancora, secondo e terzo le coordinate
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

	# print('238', y_rpn_overlap[0, :, :, :] == 1)
	num_pos = len(pos_locs[0])
	num_neg = len(neg_locs[0])

	# print('239', pos_locs)
	# print('240', num_pos)
	# print('241', num_neg)
	# print('pos_reg', np.where(y_rpn_regr[0, :, :, :] > 0))
	
	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.
	num_regions = 256

	if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		# print('249', val_locs)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2
	# else:
	# 	# Qui campioniamo num_regions/2 - len(pos_locs[0] volte ancore positive per bilanciare il batch tra pos e neg.
	# 	# Alla fine possiamo semplicemente concatenare ad y_is_box_valid e y_rpn_overlap perchè poi la loss_cls filtra le ancore non valide
	# 	new_pos = np.random.choice(len(pos_locs[0]), int(num_regions/2 - len(pos_locs[0])))

	# 	# print('new_pos:', len(new_pos), new_pos)
	# 	# print(y_is_box_valid[0, pos_locs[0][new_pos], pos_locs[1][new_pos], pos_locs[2][new_pos]])
	# 	# print(y_rpn_overlap[0, pos_locs[0][new_pos], pos_locs[1][new_pos], pos_locs[2][new_pos]])

	# 	# pos_y_is_box_valid = y_is_box_valid[0, pos_locs[0][new_pos], pos_locs[1][new_pos], pos_locs[2][new_pos]]
	# 	# pos_y_rpn_overlap = y_rpn_overlap[0, pos_locs[0][new_pos], pos_locs[1][new_pos], pos_locs[2][new_pos]]
	# 	# double_check = np.where(np.logical_and(pos_y_rpn_overlap[0, :, :, :] == 1, pos_y_is_box_valid[0, :, :, :] == 1))

	# 	# print('double check:', len(double_check[0]))
	# 	# print('pos_y_is_box_valid shape',pos_y_is_box_valid.shape)
	# 	# print('pos_y_rpn_overlap shape',pos_y_rpn_overlap.shape)

	# 	# num_pos = new_pos + num_pos
	
	# # Estendiamo y_is_box_valid e y_rpn_overlap
	# not_valids = np.where(y_is_box_valid[0, :, :, :] == 0)
	# replacing_indexes = int(num_regions/2 - len(pos_locs[0]))
	# not_valid_to_be_replaced = np.array([not_valids[0][:replacing_indexes], not_valids[1][:replacing_indexes], not_valids[2][:replacing_indexes]])

	# # print('not val', len(not_valids[0]))
	# # print('not val', not_valid_to_be_replaced.shape)
	# # print('not val', not_valid_to_be_replaced[0])
	# # print('not val', 4*not_valid_to_be_replaced[0])
	# # print('not val', 4*not_valid_to_be_replaced[0]+4)
	# # print(type(y_rpn_regr))

	# y_is_box_valid[0, not_valid_to_be_replaced[0], not_valid_to_be_replaced[1], not_valid_to_be_replaced[2]] = 1
	# y_rpn_overlap[0, not_valid_to_be_replaced[0], not_valid_to_be_replaced[1], not_valid_to_be_replaced[2]] = 1

	# for idx in range(len(not_valid_to_be_replaced[0])):
	# 	# print('?'*9)
	# 	new_regr_values = new_pos[idx]
	# 	# print(y_rpn_regr[0, 4*not_valid_to_be_replaced[0][idx]:4*not_valid_to_be_replaced[0][idx]+4, not_valid_to_be_replaced[1][idx], not_valid_to_be_replaced[2][idx]])
	# 	# print(y_rpn_regr[0, 4*pos_locs[0][new_regr_values]:4*pos_locs[0][new_regr_values]+4, pos_locs[1][new_regr_values],pos_locs[2][new_regr_values]])
	# 	y_rpn_regr[0, 4*not_valid_to_be_replaced[0][idx]:4*not_valid_to_be_replaced[0][idx]+4, not_valid_to_be_replaced[1][idx], not_valid_to_be_replaced[2][idx]] = y_rpn_regr[0, 4*pos_locs[0][new_regr_values]:4*pos_locs[0][new_regr_values]+4, pos_locs[1][new_regr_values],pos_locs[2][new_regr_values]]
	# 	# print(y_rpn_regr[0, 4*not_valid_to_be_replaced[0][idx]:4*not_valid_to_be_replaced[0][idx]+4, not_valid_to_be_replaced[1][idx], not_valid_to_be_replaced[2][idx]])
	
	# num_pos += len(new_pos)
	# # print('new num pos:', num_pos)


	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		# print('255', len(val_locs))
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

	# # debug
	# print('y_is_box_valid shape',y_is_box_valid.shape)
	# print('y_is_box_valid shape',y_is_box_valid[0,14,17,17])
	# print('y_is_box_valid shape',y_is_box_valid[0,0,0,1])
	# print('y_rpn_overlap shape',y_rpn_overlap.shape)
	# print('y_rpn_overlap shape',y_rpn_overlap[0,14,17,17])
	# print('regr:', y_rpn_regr.shape)
	# print('regr:', y_rpn_regr[0,4*14:4*14+4,17,17])
	# print('new pos locs',len(np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))[0])) #ritorna 3 array: il primo sono gli indici di ancora, secondo e terzo le coordinate
	# print('new neg locs',len(np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))[0]))


	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr), num_pos

def rescale_image(img_orig):
	img_aug = copy.copy(img_orig)

	rows, cols = img_orig.shape[:2]
	width_percent = (C.resizeFinalDim / float(cols))
	hsize = int((float(rows) * float(width_percent)))
	img = Image.fromarray(img_aug)
	img_aug = np.array(img.resize((C.resizeFinalDim , hsize), PIL.Image.ANTIALIAS))

	return img_aug, width_percent


def augment(img_path, img_data_path, augment=True, **kwargs):	
	img_orig = np.load(img_path)
	img_data_orig = pd.read_pickle(img_data_path)

	img_data_aug = copy.deepcopy(img_data_orig)

	# resize paches, the final dimension is the one set in the config (C.resizeFinalDim)
	if C.resizePatch:
		img_aug, width_percent = rescale_image(img_orig)

		for index, row in img_data_aug.iterrows():			
			x1 = row['x1s'] * width_percent
			x2 = row['x2s'] * width_percent
			y1 = row['y1s'] * width_percent
			y2 = row['y2s'] * width_percent
			img_data_aug.at[index,'x1s']= x1
			img_data_aug.at[index,'x2s']= x2
			img_data_aug.at[index,'y1s']= y1
			img_data_aug.at[index,'y2s']= y2
	else:
		img_aug = copy.copy(img_orig)

	if augment:
		rows, cols = img_aug.shape[:2]

		if C.use_horizontal_flips and kwargs['hflip']:
			img_aug = cv2.flip(img_aug, 1)
			for index, row in img_data_aug.iterrows():			
				x1 = row['x1s']
				x2 = row['x2s']
				img_data_aug.at[index,'x2s']= cols - x1
				img_data_aug.at[index,'x1s']= cols - x2

		if C.use_vertical_flips and kwargs['vflip']:
			img_aug = cv2.flip(img_aug, 0)
			for index, row in img_data_aug.iterrows():			
				y1 = row['y1s']
				y2 = row['y2s']
				img_data_aug.at[index,'y2s']= rows - y1
				img_data_aug.at[index,'y1s']= rows - y2

		if C.rot_90:
			angle = kwargs['angle'] 
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

			for index, row in img_data_aug.iterrows():
				x1 = row['x1s']
				x2 = row['x2s']
				y1 = row['y1s']
				y2 = row['y2s']
				if angle == 270:
					img_data_aug.at[index,'x1s']=  y1
					img_data_aug.at[index,'x2s']=  y2
					img_data_aug.at[index,'y1s']=  cols - x2
					img_data_aug.at[index,'y2s']=  cols - x1	
				elif angle == 180:
					img_data_aug.at[index,'x2s'] = cols - x1
					img_data_aug.at[index,'x1s'] = cols - x2
					img_data_aug.at[index,'y2s'] = rows - y1
					img_data_aug.at[index,'y1s'] = rows - y2
				elif angle == 90:
					img_data_aug.at[index,'x1s'] = rows - y2
					img_data_aug.at[index,'x2s'] = rows - y1
					img_data_aug.at[index,'y1s'] = x1
					img_data_aug.at[index,'y2s'] = x2       
				elif angle == 0:
					pass

	return img_aug, img_data_aug

def get_anchor_gt(patches_path, patch_list, backbone, mode='train', use_expander=False, infinite_loop=True, pixel_mean=None):
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
	controller=True
	while controller:

		for i, patch_id in enumerate(patch_list):
			try:

				# read in image, and optionally add augmentation
				image_path = os.path.join(patches_path, patch_id, f"{patch_id}.npy")
				image_data_path = os.path.join(patches_path, patch_id, f"{patch_id}.pkl")

				#TODO: aggiungere if backbone per manipolare i pixel

				if False:#mode == 'train':
					# print('Augmenting -- START')
					hflip = np.random.randint(0, 2)
					vflip = np.random.randint(0, 2)
					angle = np.random.choice([0,90,180,270],1)[0]

					x_img, img_data_aug  = augment(image_path, image_data_path, augment=True, hflip=hflip, vflip=vflip, angle=angle)
					# print('Augmenting -- END')
				else:
					x_img, img_data_aug  = augment(image_path, image_data_path, augment=False)

				print(x_img.shape)

				img_data_aug['x1s'] = img_data_aug['x1s'].astype(int)
				img_data_aug['x2s'] = img_data_aug['x2s'].astype(int)
				img_data_aug['y1s'] = img_data_aug['y1s'].astype(int)
				img_data_aug['y2s'] = img_data_aug['y2s'].astype(int)

				(width, height) = x_img.shape
				debug_img = x_img.copy()

				if mode=='train':
					try:
						# print('calc_rpn -- START')
						y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(img_data_aug, width, height, backbone) # es.: (1, 60, 12, 12), (1, 240, 12, 12), 64
						# print('calc_rpn -- END')

					except Exception as e:
						print('Exception in calc_rpn: ', e)
						continue
				else:
					y_rpn_cls = np.zeros((1,1,1,1))
					y_rpn_regr = np.zeros((1,1,1,1))
					num_pos = 0

				# print('starting shape: ', x_img.shape)
				if not use_expander:
					x_img = np.tile(x_img,(3,1,1))
					# x_img = np.expand_dims(x_img, axis=0) # (600, 600) --> (1, 600, 600)
					x_img = np.transpose(x_img, (1, 2, 0))
					# print('expanded shape: ', x_img.shape)
					# x_img = np.repeat(x_img, 3, axis=3)
					# print('ending shape: ', x_img.shape)

				# Zero-center by mean pixel, and preprocess image
				# print('zero-centering -- START')
				
				# zero_centering(x_img, pixel_mean)
				
				if (backbone == 'resnet50' or backbone =='baseline_16' or backbone =='baseline_44'):
					normalize_pixel_values(x_img)
				
				x_img = np.expand_dims(x_img, axis=0) # (600, 600) --> (1, 600, 600)
				if use_expander:				
					x_img = np.expand_dims(x_img, axis=3) # (1, 600, 600) --> (1, 600, 600, 1)

				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

				y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1)) # (1, 60, 12, 12) --> (1, 12, 12, 60)
				y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1)) # (1, 240, 12, 12) --> (1, 12, 12, 240)
				
				if (i==len(patch_list)-1 and not infinite_loop):
					controller=False
				
				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug, debug_img, num_pos, patch_id

			except Exception as e:
				print('Exception in get_anchor', e)
				continue


def zero_centering(img_patch, pixel_mean=None):
	if pixel_mean is None:
		img_channel_mean = C.img_channel_mean
	else:
		img_channel_mean = pixel_mean

    #in order to manage images with 3 channels. ours should be 1
	imageChannels = 1 if len(img_patch.shape)== 2 else 3
    
	if(imageChannels == 1):
		img_patch[:, :] -= img_channel_mean[0]
	else:
		# print('three channels')
		img_patch[:, :, 0] -= img_channel_mean[0]
		img_patch[:, :, 1] -= img_channel_mean[1]
		img_patch[:, :, 2] -= img_channel_mean[2]
	return

def normalize_pixel_values(img_patch):

	if False:
		img_patch[:, :, 0] /= 255
		img_patch[:, :, 1] /= 255
		img_patch[:, :, 2] /= 255
	else:
		# img_patch[:, :, 0] = img_patch[:, :, 0]/C.img_channel_mean[0]
		# img_patch[:, :, 1] = img_patch[:, :, 1]/C.img_channel_mean[1]
		# img_patch[:, :, 2] = img_patch[:, :, 2]/C.img_channel_mean[2]
		max = img_patch[:, :, 0].max()
		img_patch[:, :, 0] /= max
		img_patch[:, :, 1] /= max
		img_patch[:, :, 2] /= max

	return