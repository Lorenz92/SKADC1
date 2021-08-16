import os

# Generic config
RANDOM_SEED = 42
DATA_SUBSET = 1
DOWNLOAD_FOLDER = 'data/training'
DATA_FOLDER = os.path.join(os.getcwd(), "data")
TRAIN_DATA_FOLDER = os.path.join(DATA_FOLDER, "training")
VAL_DATA_FOLDER = os.path.join(DATA_FOLDER, "predictions")
TRAIN_PATCHES_FOLDER = os.path.join(TRAIN_DATA_FOLDER, "patches")
EVAL_RESULTS = os.path.join(VAL_DATA_FOLDER, "patches")
TRAIN_SET_PATH_CLEANED = os.path.join(TRAIN_DATA_FOLDER, "B1_training_clean_SKA.txt") #TODO: fix this
TRAIN_SET_PATH = os.path.join(TRAIN_DATA_FOLDER, "B1_training_SKA.txt") #TODO: fix this
IMAGE_PATH = os.path.join(TRAIN_DATA_FOLDER, '560Mhz_1000h.fits')
PRIMARY_BEAM_PATH = os.path.join(TRAIN_DATA_FOLDER, 'PrimaryBeam_B1.fits')

MODEL_WEIGHTS = os.path.join(os.getcwd(), "model")

required_files = [
    {
        "file_name": "B1_training_SKA.txt",
        "url": "https://owncloud.ia2.inaf.it/index.php/s/iTOVkIL6EfXkcdR/download" #54Mb
    },
    {
        "file_name": "B1_training_clean_SKA.txt",
        "url": "https://owncloud.ia2.inaf.it/index.php/s/I4HL84Etgw9c2Le/download" #54Mb
    },
    {
        "file_name": "PrimaryBeam_B1.fits",
        "url": "https://owncloud.ia2.inaf.it/index.php/s/ZbaSDe7zGBYgxL1/download" #300Kb
    },
    # {
    #     "file_name": "SynthesBeam_B1.fits",
    #     "url": "https://owncloud.ia2.inaf.it/index.php/s/cwzf1BO2pyg9TVv/download" #4Gb
    # },
    {
        "file_name": "560Mhz_1000h.fits",
        "url": "https://owncloud.ia2.inaf.it/index.php/s/hbasFhd4YILNkCr/download" #4Gb
    },
    #     {
    #     "file_name": "B5_training_SKA.txt",
    #     "url": "https://owncloud.ia2.inaf.it/index.php/s/Y5CIa5V3QiBu1M1/download" #54Mb
    # },
    # {
    #     "file_name": "9200Mhz_1000h.fits",
    #     "url":"https://owncloud.ia2.inaf.it/index.php/s/nK8Pqf3XIaXFuKD/download"
    # }
]

# Dimensions of patches
patch_dim = None

# Setting for data augmentation
use_horizontal_flips = True
use_vertical_flips = True
rot_90 = True

# Anchor box scales
# Note that if im_size is smaller, anchor_box_scales should be scaled
# Original anchor_box_scales in the paper is [128, 256, 512]
anchor_box_scales = None

# Anchor box ratios
# anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
anchor_box_ratios = [[2, 1], [1, 2], [1,1]]
# anchor_box_ratios = [[2, 1]]

anchor_num = None

#NMS max boxes
nms_max_boxes = 2000

#convert from float (0,1) images to int(0, 255)
convert_to_RGB = True 

# image channel-wise mean to subtract
img_channel_mean = [123.68, 116.779, 103.939]
#img_channel_max = 0.006585681
#img_channel_min = 0.0

#img_channel_mean =  [2.1317146e-06] #mean value of all patches
img_scaling_factor = 1.0

# This is used in order to resize the biggest bbox to match the network input shape
resize_image_factor = 1.

# Control wheter to remove bbox with area less than resize_image_factor
clean_dataset = True

# Control bbox enlargement
bbox_scale_factor = 3
enlarge_bbox = True

# Stretching parameter for the gamma function
gamma = 0.7

# resize the original patch
resizePatch = True

# final dimension of the patch
resizeFinalDim = None

# number of ROIs at once
num_rois = None

# stride at the RPN (this depends on the network configuration)
# rpn_stride = 16

# scaling the stdev
std_scaling = 4.0
classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
# classifier_regr_std = [10.0, 10.0, 5.0, 5.0] #ICRAR

pooling_regions = 7
num_rois = 4

# overlaps for RPN
rpn_min_overlap = 0.3
rpn_max_overlap = 0.7

# overlaps for classifier ROIs
classifier_min_overlap = 0.1
classifier_max_overlap = 0.5