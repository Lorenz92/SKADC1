{
  "nbformat": 4,
  "nbformat_minor": 2,
  "metadata": {
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.8"
    },
    "orig_nbformat": 2,
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3.8.8 64-bit"
    },
    "colab": {
      "name": "main.ipynb",
      "provenance": []
    },
    "metadata": {
      "interpreter": {
        "hash": "31f2aee4e71d21fbe5cf8b01ff0e069b9275f58929596ceb00d14d90e3e16cd6"
      }
    },
    "interpreter": {
      "hash": "2d04b9be2429acaaca541b9fd5f04e1d13e142a6d8a3b01163614a22cbe6d5d6"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# !git clone 'https://github.com/Lorenz92/SKADC1.git'\r\n",
        "# % cd SKADC1\r\n",
        "# !echo $PWD"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "import pandas as pd\r\n",
        "import numpy as np\r\n",
        "\r\n",
        "import src.dataset as dataset\r\n",
        "import src.config as config \r\n",
        "from src.utils import *\r\n",
        "import src.models as models\r\n",
        "import src.losses as loss\r\n",
        "\r\n",
        "path = config.TRAIN_PATCHES_FOLDER\r\n",
        "%load_ext autoreload\r\n",
        "%autoreload 2\r\n",
        "\r\n",
        "np.random.seed(config.RANDOM_SEED)"
      ],
      "outputs": [],
      "metadata": {
        "id": "7WwOlsifF-5G",
        "outputId": "303002ab-c8c4-4807-caeb-ee1c590d4326",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "if 'google.colab' in str(get_ipython()):\r\n",
        "  use_colab = True\r\n",
        "  print('Running on CoLab')\r\n",
        "else:\r\n",
        "  use_colab = False\r\n",
        "  print('Not running on CoLab')"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "if use_colab:\r\n",
        "    # Read file from Colab Notebook\r\n",
        "    from google.colab import drive\r\n",
        "    drive.mount('/content/drive')\r\n",
        "    config.MODEL_WEIGHTS = \"/content/drive/My Drive/Colab Notebooks/SKADC1\"\r\n",
        "    config.IMAGE_PATH = \"/content/drive/My Drive/Colab Notebooks/SKADC1/asset/560Mhz_1000h.fits\""
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Dataset parsing and loading\r\n",
        "# use \"subset\" in config file to load a small portion of data for development/debugging purposes\r\n",
        "ska_dataset = dataset.SKADataset(k=3, print_info=False, use_pb=False)"
      ],
      "outputs": [],
      "metadata": {
        "tags": [],
        "id": "Q8IbYCwkF-5K",
        "outputId": "5cd8e5ba-49cd-49b4-d65c-65d82229f792",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 221
        }
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "ska_dataset.cleaned_train_df[['width', 'height', 'area_orig', 'area_cropped']].describe()"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "ska_dataset.cleaned_train_df[['width', 'height', 'area_orig']].quantile([.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98,.99,1.])"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "show_plot = True\r\n",
        "RGB_patch_norm = True\r\n",
        "ska_dataset.generate_patches(limit=2, patch_RGB_norm =  RGB_patch_norm, plot_patches=show_plot) # Remember to remove internal return"
      ],
      "outputs": [],
      "metadata": {
        "id": "Ug10Ku-OF-5Y",
        "outputId": "c3b6bed6-24a7-4f71-cbed-c4b6b8d934af",
        "tags": []
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "ska_dataset.analyze_class_distribution()"
      ],
      "outputs": [],
      "metadata": {
        "tags": []
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "ska_dataset.split_train_val_stratified(random_state=42, val_portion=0.2)\r\n",
        "\r\n",
        "print(len(ska_dataset.train_patch_list))\r\n",
        "print(len(ska_dataset.val_patch_list))\r\n"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "ska_dataset.split_train_val(random_state=42, val_portion=0.2)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "## datagen + calc_rpn -> rpn_net -> rpn_to_roi -> calc_iou -> cls_net"
      ],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Training"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "print_img(config.TRAIN_PATCHES_FOLDER, '20_16396_16729_20', show_data=True)\r\n"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Debugging\r\n",
        "\r\n",
        "train_patch_list = ska_dataset.train_patch_list\r\n",
        "patches_folder_path=config.TRAIN_PATCHES_FOLDER\r\n",
        "\r\n",
        "train_datagen = prep.get_anchor_gt(patches_folder_path, ['20_16396_16729_20'], backbone, pixel_mean=None)\r\n",
        "image, [y_rpn_cls_true, y_rpn_reg_true], img_data_aug, _, _, patch_id = next(train_datagen)\r\n",
        "\r\n"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Get FRCNN model"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Choose the feature extraction model\n",
        "backbone='baseline_16'\n",
        "# backbone='vgg16'\n",
        "\n",
        "if backbone=='baseline_16':\n",
        "    config.resizePatch = True\n",
        "    config.rpn_stride = 4\n",
        "    config.num_rois = 128\n",
        "    # config.anchor_box_scales = [16,32,64] # anchors dim in the last feature map, 25x25\n",
        "    config.anchor_box_scales = [6, 8, 12, 16, 24]\n",
        "    config.resizeFinalDim = 100\n",
        "    input_shape_1 = config.resizeFinalDim\n",
        "else:\n",
        "    config.resizePatch = True\n",
        "    config.rpn_stride = 16\n",
        "    config.num_rois = 4\n",
        "    config.resizeFinalDim = 600\n",
        "    input_shape_1=config.resizeFinalDim\n",
        "    config.anchor_box_scales = [32, 64, 128]\n",
        "\n",
        "\n",
        "config.anchor_num = len(config.anchor_box_ratios)*len(config.anchor_box_scales)\n",
        "input_shape_2=(None, 4)\n",
        "\n",
        "print(config.resizePatch)\n",
        "print(config.rpn_stride)\n",
        "\n",
        "checkpoint = get_model_last_checkpoint(backbone)\n",
        "print(f'Model last checkpoint: {checkpoint}')"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "rpn_model, detector_model, total_model = models.get_train_model(input_shape_1=input_shape_1, input_shape_2=input_shape_2, anchor_num=config.anchor_num, pooling_regions=config.pooling_regions, num_rois=config.num_rois, num_classes=len(ska_dataset.class_list)+1, backbone=backbone, use_expander=False)\r\n",
        "\r\n",
        "rpn_model.summary()\r\n",
        "detector_model.summary()\r\n",
        "total_model.summary()"
      ],
      "outputs": [],
      "metadata": {
        "tags": [
          "outputPrepend"
        ]
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Load weights"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "models.load_weigths(rpn_model, detector_model, backbone, resume_train=False, checkpoint=checkpoint)\r\n",
        "models.compile_models(rpn_model, detector_model, total_model, rpn_losses=[loss.rpn_loss_cls, loss.rpn_loss_regr], detector_losses=[loss.detector_loss_cls, loss.detector_loss_regr], class_list=ska_dataset.class_list)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Specifically checking backbone weights\r\n",
        "\r\n",
        "# total_model.weights[24:25][0][0][0][0]"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# Check that all of the pretrained weights have been loaded.\r\n",
        "import numpy as np\r\n",
        "for i, j in zip(total_model.weights, rpn_model.weights): \r\n",
        "    # print(i,j)\r\n",
        "    assert np.allclose(i,j), 'Weights don\\'t match!'"
      ],
      "outputs": [],
      "metadata": {
        "tags": []
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Train"
      ],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "- errore \"Exception: 'a' cannot be empty unless no samples are taken\" quando nel sampling ci sono meno di 4 roi\n",
        "- errore \"None type object is not iterable\" dovuto al max(IoUs) quando calc_iou torna None, None, None, None\n",
        "- patch di debug 1550_16376_16779_100\n",
        "- capire il parametro classifier_regr_std in che modo influenza il training"
      ],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "- provare normalizzazione valori immagini di input\n",
        "- provare a far passare più roi anzichè 4"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "# zzz = np.load(f'{config.TRAIN_PATCHES_FOLDER}/3100_16376_16829_100/3100_16376_16829_100.npy')\r\n",
        "\r\n",
        "# print_img(config.TRAIN_PATCHES_FOLDER, '3100_16376_16829_100')"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "20_16396_16729_20"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "from src.train import *\r\n",
        "val_patch_list = None\r\n",
        "train_frcnn(rpn_model, detector_model, total_model, ['20_16396_16729_20'], ska_dataset.val_patch_list, ska_dataset.class_list, num_epochs=500, patches_folder_path=config.TRAIN_PATCHES_FOLDER, single_patch_norm = RGB_patch_norm, backbone=backbone, resume_train=True)\r\n",
        "# train_frcnn(rpn_model, detector_model, total_model, ska_dataset.train_patch_list, ska_dataset.val_patch_list, ska_dataset.class_list, num_epochs=300, patches_folder_path=config.TRAIN_PATCHES_FOLDER, backbone=backbone, resume_train=True)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": 59,
      "source": [
        "total_model.save_weights(f'{config.MODEL_WEIGHTS}/{backbone}/0_frcnn_{backbone}.h5')"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Validation"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": 60,
      "source": [
        "rpn_model_eval, detector_model_eval, total_model_eval = models.get_eval_model(input_shape_1=input_shape_1, input_shape_2=input_shape_2, input_shape_fmap=None, anchor_num=config.anchor_num, pooling_regions=config.pooling_regions, num_rois=config.num_rois, num_classes=len(ska_dataset.class_list)+1, backbone=backbone, use_expander=False)\r\n",
        "\r\n",
        "rpn_model_eval.summary()\r\n",
        "detector_model_eval.summary()\r\n",
        "total_model_eval.summary()"
      ],
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"RegionProposal\"\n",
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "input_1 (InputLayer)         [(None, 100, 100, 3)]     0         \n",
            "_________________________________________________________________\n",
            "block1_conv1 (Conv2D)        (None, 100, 100, 64)      1792      \n",
            "_________________________________________________________________\n",
            "block1_conv2 (Conv2D)        (None, 100, 100, 64)      36928     \n",
            "_________________________________________________________________\n",
            "block1_pool (MaxPooling2D)   (None, 50, 50, 64)        0         \n",
            "_________________________________________________________________\n",
            "block2_conv1 (Conv2D)        (None, 50, 50, 128)       73856     \n",
            "_________________________________________________________________\n",
            "block2_conv2 (Conv2D)        (None, 50, 50, 128)       147584    \n",
            "_________________________________________________________________\n",
            "block2_pool (MaxPooling2D)   (None, 25, 25, 128)       0         \n",
            "_________________________________________________________________\n",
            "rpn (RpnNet)                 [(None, 25, 25, 18), (Non 636506    \n",
            "=================================================================\n",
            "Total params: 896,666\n",
            "Trainable params: 636,506\n",
            "Non-trainable params: 260,160\n",
            "_________________________________________________________________\n",
            "Model: \"DetectorClassifier\"\n",
            "__________________________________________________________________________________________________\n",
            "Layer (type)                    Output Shape         Param #     Connected to                     \n",
            "==================================================================================================\n",
            "Input_fmap (InputLayer)         [(None, 25, 25, 128) 0                                            \n",
            "__________________________________________________________________________________________________\n",
            "input_2 (InputLayer)            [(None, None, 4)]    0                                            \n",
            "__________________________________________________________________________________________________\n",
            "roi_pooling (RoiPoolingConv)    (1, 128, 7, 7, 128)  0           Input_fmap[0][0]                 \n",
            "                                                                 input_2[0][0]                    \n",
            "__________________________________________________________________________________________________\n",
            "cls (Detector)                  [(1, 128, 4), (1, 12 42541072    roi_pooling[0][0]                \n",
            "==================================================================================================\n",
            "Total params: 42,541,072\n",
            "Trainable params: 42,541,072\n",
            "Non-trainable params: 0\n",
            "__________________________________________________________________________________________________\n",
            "Model: \"End2end_model\"\n",
            "__________________________________________________________________________________________________\n",
            "Layer (type)                    Output Shape         Param #     Connected to                     \n",
            "==================================================================================================\n",
            "input_1 (InputLayer)            [(None, 100, 100, 3) 0                                            \n",
            "__________________________________________________________________________________________________\n",
            "block1_conv1 (Conv2D)           (None, 100, 100, 64) 1792        input_1[0][0]                    \n",
            "__________________________________________________________________________________________________\n",
            "block1_conv2 (Conv2D)           (None, 100, 100, 64) 36928       block1_conv1[0][0]               \n",
            "__________________________________________________________________________________________________\n",
            "block1_pool (MaxPooling2D)      (None, 50, 50, 64)   0           block1_conv2[0][0]               \n",
            "__________________________________________________________________________________________________\n",
            "block2_conv1 (Conv2D)           (None, 50, 50, 128)  73856       block1_pool[0][0]                \n",
            "__________________________________________________________________________________________________\n",
            "block2_conv2 (Conv2D)           (None, 50, 50, 128)  147584      block2_conv1[0][0]               \n",
            "__________________________________________________________________________________________________\n",
            "Input_fmap (InputLayer)         [(None, 25, 25, 128) 0                                            \n",
            "__________________________________________________________________________________________________\n",
            "input_2 (InputLayer)            [(None, None, 4)]    0                                            \n",
            "__________________________________________________________________________________________________\n",
            "block2_pool (MaxPooling2D)      (None, 25, 25, 128)  0           block2_conv2[0][0]               \n",
            "__________________________________________________________________________________________________\n",
            "roi_pooling (RoiPoolingConv)    (1, 128, 7, 7, 128)  0           Input_fmap[0][0]                 \n",
            "                                                                 input_2[0][0]                    \n",
            "__________________________________________________________________________________________________\n",
            "rpn (RpnNet)                    [(None, 25, 25, 18), 636506      block2_pool[0][0]                \n",
            "__________________________________________________________________________________________________\n",
            "cls (Detector)                  [(1, 128, 4), (1, 12 42541072    roi_pooling[0][0]                \n",
            "==================================================================================================\n",
            "Total params: 43,437,738\n",
            "Trainable params: 43,177,578\n",
            "Non-trainable params: 260,160\n",
            "__________________________________________________________________________________________________\n"
          ]
        }
      ],
      "metadata": {
        "tags": [
          "outputPrepend"
        ]
      }
    },
    {
      "cell_type": "code",
      "execution_count": 61,
      "source": [
        "# Models used for mAP eval\n",
        "cp = '122_frcnn_baseline_16.h5'\n",
        "models.load_weigths(rpn_model_eval, detector_model_eval, backbone, checkpoint=cp)\n",
        "models.compile_models(rpn_model_eval, detector_model_eval, total_model_eval, rpn_losses=[loss.rpn_loss_cls, loss.rpn_loss_regr], detector_losses=[loss.detector_loss_cls, loss.detector_loss_regr], class_list=ska_dataset.class_list)"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": 67,
      "source": [
        "patch_id = ['20_16396_16729_20']#ska_dataset.train_patch_list#[1:2]\r\n",
        "# print(patch_id)\r\n",
        "# gt = pd.read_pickle(f'{config.TRAIN_PATCHES_FOLDER}/{patch_id[0]}/{patch_id[0]}.pkl')\r\n",
        "# display(gt['class_label'])\r\n",
        "\r\n",
        "preds, mAP, prec, recall = evaluate_model(rpn_model_eval, detector_model_eval, backbone, patch_id, ska_dataset.class_list, metric_threshold=.5)"
      ],
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(100, 100)\n",
            "{0: '2_2', 1: '2_3', 2: '1_1', 3: 'bg'}\n",
            "Predict\n",
            "rpn_to_roi\n",
            "[[[0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [4.8815527e-05 8.5539299e-01 4.8815527e-05 1.4450936e-01]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]\n",
            "  [0.0000000e+00 0.0000000e+00 0.0000000e+00 1.0000000e+00]]]\n",
            "[[[0. 0. 0. ... 0. 0. 0.]\n",
            "  [0. 0. 0. ... 0. 0. 0.]\n",
            "  [0. 0. 0. ... 0. 0. 0.]\n",
            "  ...\n",
            "  [0. 0. 0. ... 0. 0. 0.]\n",
            "  [0. 0. 0. ... 0. 0. 0.]\n",
            "  [0. 0. 0. ... 0. 0. 0.]]]\n",
            "Elapsed:0.1504530906677246\n",
            "(48, 56, 88, 88)\n",
            "\n",
            "Total model metrics: mAP: 0.75 - macro_precision: 0.0 - macro_recall: 0.0\n"
          ]
        }
      ],
      "metadata": {
        "tags": []
      }
    },
    {
      "cell_type": "code",
      "execution_count": 63,
      "source": [
        "preds"
      ],
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'20_16396_16729_20': {'bboxes': {'2_3': [[48.0, 56.0, 88.0, 88.0]]},\n",
              "  'probs': {'2_3': [0.855393]},\n",
              "  'mAP': 0.75,\n",
              "  'macro_precision': 0.0,\n",
              "  'macro_recall': 0.0}}"
            ]
          },
          "metadata": {},
          "execution_count": 63
        }
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": 64,
      "source": [
        "print(config.patch_dim , float(config.resizeFinalDim))"
      ],
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "20 100.0\n"
          ]
        }
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": 65,
      "source": [
        "print_img(config.TRAIN_PATCHES_FOLDER, '20_16396_16729_20')"
      ],
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAQUAAAD4CAYAAADl7fPiAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAQZElEQVR4nO3df6zV9X3H8eeLc++FwrCVolSFWdMyGzSRNQTXzS04W4fEjnbpKqRxtDXBNTVZky2b3RJtun/aLM5s0WhpS1DTKs02WpIyhbgl1qxa0aBCxUgJhntFrkoF7gQu59z3/jjf293P5Rz43vM9v+7h9UjI+X4/38/5fj/fXHjd748Pn48iAjOzcTM63QAz6y4OBTNLOBTMLOFQMLOEQ8HMEn2dbkAtA5oZs5jT6WaYTY2Uv2r/FP7pTWG/jOV7m3iicozRyomaO+7KUJjFHK7VDZ1uhtmUqH8gd93SpQty142B/vxtOHEqV73/efPRutt8+2BmiUKhIGmlpFcl7ZN0Z43tMyVtzrY/K+nDRY5nZq3XcChIKgH3AzcBS4C1kpZMqnYb8OuI+ChwL/DtRo9nZu1R5EphObAvIvZHxCjwGLB6Up3VwEPZ8r8BN0hTeWpiZu1WJBQuAw5OWB/MymrWiYgycBT4YK2dSVovaaeknafJ97DEzJqvax40RsSGiFgWEcv6mdnp5pidt4qEwhCwaML6wqysZh1JfcD7gXcKHNPMWqxIKDwHLJZ0haQBYA2wdVKdrcC6bPlzwH+F/6+2WVdruPNSRJQl3QE8AZSAjRGxR9I3gZ0RsRX4PvCIpH3AEarBYWZdrFCPxojYBmybVHbXhOWTwJ8XOYbZdBGnR/NXLpXy73f2FJ6x5e39+Fb9m4SuedBoZt3BoWBmCYeCmSUcCmaWcCiYWcKhYGYJh4KZJRwKZpZwKJhZwqFgZomuHLjVrNdVDr6Ru25pwUVT2HGlcD1fKZhZwqFgZgmHgpklHApmlnAomFnCoWBmCYeCmSWKzBC1SNJ/S/qlpD2S/qpGnRWSjkralf25q9a+zKx7FOm8VAb+OiJekDQXeF7Sjoj45aR6P4uImwscx8zaqOErhYg4FBEvZMvHgVc4c4YoM5tmmtLNOZtN+neBZ2ts/oSkF4E3gL+JiD119rEeWA8wi9nNaJZZ15rKyM/lwclzLDXh+FGuu61wKEj6LeDfga9FxLFJm18ALo+IEUmrgB8Di2s3MjYAGwAu0DxPGGPWIYXePkjqpxoIP4iI/5i8PSKORcRItrwN6Jc0v8gxzay1irx9ENUZoF6JiH+uU+dD41PPS1qeHc9zSZp1sSK3D38A3Aq8LGlXVvb3wG8DRMSDVOeP/IqkMnACWOO5JM26W5G5JJ8GdI469wH3NXoMM2s/92g0s4RDwcwSDgUzSzgUzCzhUDCzhEdzbied9WVNym9urUN8pWBmCYeCmSUcCmaWcCiYWcKhYGYJh4KZJRwKZpZwKJhZwqFgZgn3aGyj0vz8I9FV3nqrhS0xq89XCmaWcCiYWaJwKEg6IOnlbFq4nTW2S9K/Ston6SVJHy96TDNrnWY9U7g+It6us+0mqnM9LAauBR7IPs2sC7Xj9mE18HBUPQN8QNIlbTiumTWgGaEQwHZJz2dTv012GXBwwvogNeaclLRe0k5JO09zqgnNMrNGNOP24bqIGJJ0MbBD0t6IeGqqO/G0cWbdofCVQkQMZZ/DwBZg+aQqQ8CiCesLszIz60JF55KcI2nu+DJwI7B7UrWtwF9kbyF+DzgaEYeKHNfMWqfo7cMCYEs2XWQf8MOIeFzSX8Jvpo7bBqwC9gHvAV8qeEwza6FCoRAR+4FrapQ/OGE5gK8WOU7Xm1HKVS1OnmxxQ8yKc49GM0s4FMws4VAws4RDwcwSDgUzSzgUzCzhUDCzhEPBzBIOBTNLOBTMLOHRnJthrJKv2vHjLW6IWXG+UjCzhEPBzBIOBTNLOBTMLOFQMLOEQ8HMEg4FM0s0HAqSrsymihv/c0zS1ybVWSHp6IQ6dxVusZm1VMOdlyLiVWApgKQS1WHbt9So+rOIuLnR45hZezXr9uEG4FcR8XqT9mdmHdKsUFgDPFpn2yckvSjpPyVdVW8HnjbOrDuoOgJ7gR1IA8AbwFURcXjStguAsYgYkbQK+JeIWHyufV6geXGtbijULjOr79l4kmNxRLW2NeNK4SbghcmBABARxyJiJFveBvRLmt+EY5pZizQjFNZS59ZB0oeUTR8laXl2vHeacEwza5FC/3U6mz/yU8DtE8omThn3OeArksrACWBNFL1fMbOWKvxMoRX8TMGstVr9TMHMeohDwcwSDgUzSzgUzCzhUDCzhEPBzBIOBTNLOBTMLOFQMLOEQ8HMEg4FM0s4FMws4VAws4RDwcwSDgUzSzgUzCzhUDCzhEPBzBK5QkHSRknDknZPKJsnaYek17LPC+t8d11W5zVJ65rVcDNrjbxXCpuAlZPK7gSezOZxeDJbT0iaB9wNXAssB+6uFx5m1h1yhUJEPAUcmVS8GngoW34I+EyNr/4JsCMijkTEr4EdnBkuZtZFijxTWBARh7LlN4EFNepcBhycsD6YlZlZl2rKg8ZsLodCY8V7Lkmz7lAkFA5LugQg+xyuUWcIWDRhfWFWdoaI2BARyyJiWT8zCzTLzIooEgpbgfG3CeuAn9So8wRwo6QLsweMN2ZlZtal8r6SfBT4OXClpEFJtwHfAj4l6TXgk9k6kpZJ+h5ARBwB/hF4LvvzzazMzLqUp40zOw952jgzy82hYGYJh4KZJRwKZpZwKJhZwqFgZgmHgpklHApmlnAomFnCoWBmCYeCmSX6Ot0As7NRX/6/olEut7Al5w+HghVyS+xlgErL9q9K/ovZiLGWtaPZRimxWR/rdDNqcihYIQNUeERXtWz/Kk3hSiGmz5XCrbGn002oy88UzCzhUDCzhEPBzBIOBTNLOBTMLHHOR7uSNgI3A8MRcXVW9k/Ap4FR4FfAlyLi3RrfPQAcBypAOSKWNa3l1vNuib3MrOQfQ3S6vJIcpdTpJpxVnvc9m4D7gIcnlO0Avh4RZUnfBr4O/F2d718fEW8XaqWdlwao8Ejpmtz1p8sryW5+HQk5bh9qzSMZEdvj/38Cz1Cd5MXMekAzOi99GdhcZ1sA2yUF8J2I2FBvJ5LWA+sBZjG7Cc2ybjXj6nw9+XT4MGO/c3Xu/caMmiOW1953Od+tRv/ho7n3Wd5/IHfdblYoFCT9A1AGflCnynURMSTpYmCHpL3ZlccZssDYANV5H4q0y8wa1/DbB0lfpPoA8gtRZ0aZiBjKPoeBLcDyRo9nZu3RUChIWgn8LfCnEfFenTpzJM0dX6Y6j+TuRhtqZu1xzlCoM4/kfcBcqrcEuyQ9mNW9VNK27KsLgKclvQj8AvhpRDzekrMws6Y55zOFiFhbo/j7deq+AazKlvcD+d8nmVlXcI9GM0s4FMws4VAws4RDwcwSDgUzS3iMRmuK0vwP5q576uI5uepVRgYYWTgr934rM6fQzTnn/7583wX9ufc5+8TJXPVmHHudsZGR3PttN18pmFnCoWBmCYeCmSUcCmaWcCiYWcKhYGYJh4KZJRwKZpZwKJhZwj0arTmU//dLZWa+ulESp+fk76VYnj2VgVvz1et/bwpzNMyama/ee939z85XCmaWcCiYWSLPGI0bJQ1L2j2h7BuShrLxGXdJWlXnuyslvSppn6Q7m9lwM2uNPFcKm4CVNcrvjYil2Z9tkzdKKgH3AzcBS4C1kpYUaayZtV5D08bltBzYFxH7I2IUeAxY3cB+zKyNijxTuEPSS9ntxYU1tl8GHJywPpiV1SRpvaSdknae5lSBZplZEY2GwgPAR4ClwCHgnqINiYgNEbEsIpb1k/PVjpk1XUOhEBGHI6ISEWPAd6k9HdwQsGjC+sKszMy6WKPTxl0yYfWz1J4O7jlgsaQrJA0Aa4CtjRzPzNrnnF2rsmnjVgDzJQ0CdwMrJC2lOtX8AeD2rO6lwPciYlVElCXdATwBlICNEbGnFSdhZs3TsmnjsvVtwBmvK60Hncr/cLh0aixXPVUCVRpt0Dn2nW/cVmaUc1YE4njOwVhHR3PvsxPco9HMEg4FM0s4FMws4VAws4RDwcwSDgUzSzgUzCzhUDCzhEPBzBIOBTNLdPewsjZtVI4dy1135uF83YH7RkaZ9e4U+jkr/8jLytfTOneXbIA4la/7cpRb1He7SXylYGYJh4KZJRwKZpZwKJhZwqFgZgmHgpklHApmlsgzRuNG4GZgOCKuzso2A1dmVT4AvBsRS2t89wBwHKgA5YhY1pRW23lhdEYfX35le+765fdN4XdczlHWBkby9ykonXwnV71RlejPvdf2y9N5aRNwH/DweEFE3DK+LOke4OhZvn99RLzdaAPt/PWji36fkY++P3f9kxc2v/PSnDdP597nzLdezV33C6eeyV233fIM3PqUpA/X2iZJwOeBP25yu8ysQ4p2c/5D4HBEvFZnewDbJQXwnYjYUG9HktYD6wFmMbtgs6ybVfbk/406p/Sx3HVLp+bmrjvWp1z1+v63nHuf1d+R01/RUFgLPHqW7ddFxJCki4EdkvZmE9aeIQuMDQAXaF7+cbXNrKkafvsgqQ/4M2BzvToRMZR9DgNbqD29nJl1kSKvJD8J7I2IwVobJc2RNHd8GbiR2tPLmVkXOWcoZNPG/Ry4UtKgpNuyTWuYdOsg6VJJ4zNCLQCelvQi8AvgpxHxePOabmat0Oi0cUTEF2uU/WbauIjYD1xTsH1m1mbu0WhmCYeCmSUcCmaWcCiYWcKhYGYJj+ZsXW3spb25684+fnnuujF7Vr6KOUdoBhg7cTJ33W7mKwUzSzgUzCzhUDCzhEPBzBJ+0GiFjFLi1tjT6WYAMOPIUP7Kx3P+1a/kH44tyvmnzjtF/lGi2s2hYIVsVv5BUFqtb16H3z6M5A+lOJ1/v+3m2wczSzgUzCzhUDCzhEPBzBKK6L4xUiW9Bbw+qXg+0IvzR/TqeUHvnlsvnNflEXFRrQ1dGQq1SNrZizNM9ep5Qe+eW6+e1zjfPphZwqFgZonpFAp1Z5ea5nr1vKB3z61XzwuYRs8UzKw9ptOVgpm1gUPBzBLTIhQkrZT0qqR9ku7sdHuaRdIBSS9L2iVpZ6fbU4SkjZKGJe2eUDZP0g5Jr2WfF3ayjY2oc17fkDSU/dx2SVrVyTY2W9eHgqQScD9wE7AEWCtpSWdb1VTXR8TSHnjvvQlYOansTuDJiFgMPJmtTzebOPO8AO7Nfm5LI2Jbje3TVteHAtWZqvdFxP6IGAUeA1Z3uE02SUQ8BRyZVLwaeChbfgj4TDvb1Ax1zqunTYdQuAw4OGF9MCvrBQFsl/S8pPWdbkwLLIiIQ9nym1QnHe4Vd0h6Kbu9mHa3RWczHUKhl10XER+nemv0VUl/1OkGtUpU3333yvvvB4CPAEuBQ8A9HW1Nk02HUBgCFk1YX5iVTXsRMZR9DgNbqN4q9ZLDki4ByD6HO9yepoiIwxFRiYgx4Lv02M9tOoTCc8BiSVdIGgDWAFs73KbCJM2RNHd8GbgR2H32b007W4F12fI64CcdbEvTjAdd5rP02M+t68dojIiypDuAJ4ASsDGiS0YKLWYBsEUSVH8OP4yIxzvbpMZJehRYAcyXNAjcDXwL+JGk26j+V/jPd66FjalzXiskLaV6O3QAuL1T7WsFd3M2s8R0uH0wszZyKJhZwqFgZgmHgpklHApmlnAomFnCoWBmif8DmHWgYuyaJJkAAAAASUVORK5CYII="
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": 68,
      "source": [
        "anch = pd.read_pickle(f'{config.EVAL_RESULTS}/20_16396_16729_20/20_16396_16729_20.pkl')\r\n",
        "display(anch)"
      ],
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "   x1s  y1s  x2s  y2s class      prob\n",
              "0    9   11   17   17   2_3  0.855393"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>x1s</th>\n",
              "      <th>y1s</th>\n",
              "      <th>x2s</th>\n",
              "      <th>y2s</th>\n",
              "      <th>class</th>\n",
              "      <th>prob</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9</td>\n",
              "      <td>11</td>\n",
              "      <td>17</td>\n",
              "      <td>17</td>\n",
              "      <td>2_3</td>\n",
              "      <td>0.855393</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ]
          },
          "metadata": {}
        }
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": 69,
      "source": [
        "anch['width'] = anch['x2s'] - anch['x1s']\r\n",
        "anch['heght'] = anch['y2s'] - anch['y1s']"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "anch.describe() "
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": 70,
      "source": [
        "print_img(config.TRAIN_PATCHES_FOLDER, '20_16396_16729_20', config.EVAL_RESULTS)"
      ],
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAQUAAAD4CAYAAADl7fPiAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAPyElEQVR4nO3dfYxldX3H8feHYXdxt6AgsvJUNUowaOrWbKCmtIFiKRBStLEKaSy2JGuNJDVp09A2EWP/sWmsaYNRVyWgUaRpSyVxC2xoEzT1gZXwqFC2BMMusItieaiwuzP77R9z1sxv9g6cmXtn5s74fiWbe+4533vO72bgM+dpzjdVhSQdcsRyD0DSeDEUJDUMBUkNQ0FSw1CQ1DhyuQcwyNqsq6PYsNzDkOYn6V+6Zh7/681jvRzsdzXxhaln2T/1wsAVj2UoHMUGzsp5yz0MaV6yZm3v2omTNvaurbVr+o/hhX296v7ryRvmXObhg6TGUKGQ5IIkDyXZmeSqAcvXJbmxW/7dJK8fZnuSFt+CQyHJBPBp4ELgDOCyJGfMKrsC+GlVvQn4FPC3C92epKUxzJ7CmcDOqnqkqvYDXwMumVVzCXB9N/3PwHnJfM6aSFpqw4TCycBjM97v6uYNrKmqSeAZ4NWDVpZkS5IdSXYcoN/JEkmjNzYnGqtqa1VtrqrNa1i33MORfmENEwq7gVNnvD+lmzewJsmRwCuBnwyxTUmLbJhQuBM4LckbkqwFLgVunlVzM3B5N/0e4D/Kv9WWxtqCb16qqskkVwK3AhPAtVX1QJKPAzuq6mbgi8CXk+wEnmY6OCSNsaHuaKyqbcC2WfM+OmP6ReD3h9mGtFLUgf39iycm+q93/TzOsfW9+/GpuQ8SxuZEo6TxYChIahgKkhqGgqSGoSCpYShIahgKkhqGgqSGoSCpYShIaozlg1ul1W7qscd7105sfM08Vjw1dJ17CpIahoKkhqEgqWEoSGoYCpIahoKkhqEgqTFMh6hTk/xnkh8keSDJnw6oOSfJM0nu7v59dNC6JI2PYW5emgT+rKruSnI08P0k26vqB7PqvllVFw+xHUlLaMF7ClX1RFXd1U0/B/yQwztESVphRnKbc9dN+leB7w5Y/I4k9wCPA39eVQ/MsY4twBaAo1g/imFJY2s+T36e3DW7x9IItl+Tcy4bOhSS/BLwL8BHqurZWYvvAl5XVc8nuQj4N+C0wYOsrcBWgGNynA1jpGUy1NWHJGuYDoSvVNW/zl5eVc9W1fPd9DZgTZLjh9mmpMU1zNWHMN0B6odV9fdz1Lz2UOv5JGd227OXpDTGhjl8+HXg/cB9Se7u5v0V8MsAVfVZpvtHfijJJPACcKm9JKXxNkwvyW8BeZmaa4BrFroNSUvPOxolNQwFSQ1DQVLDUJDUMBQkNXya81LKS16saXnlVsvEPQVJDUNBUsNQkNQwFCQ1DAVJDUNBUsNQkNQwFCQ1DAVJDe9oXEITx/d/Et3UU08t4kikubmnIKlhKEhqDB0KSR5Ncl/XFm7HgOVJ8o9Jdia5N8nbh92mpMUzqnMK51bVj+dYdiHTvR5OA84CPtO9ShpDS3H4cAnwpZr2HeBVSU5cgu1KWoBRhEIBtyX5ftf6bbaTgcdmvN/FgJ6TSbYk2ZFkxwH2jWBYkhZiFIcPZ1fV7iQnANuTPFhVd8x3JbaNk8bD0HsKVbW7e90L3AScOatkN3DqjPendPMkjaFhe0luSHL0oWngfOD+WWU3A3/YXYX4NeCZqnpimO1KWjzDHj5sBG7q2kUeCXy1qm5J8ifw89Zx24CLgJ3Az4A/GnKbkhbRUKFQVY8Abxsw/7Mzpgv48DDbGXtHTPQqqxdfXOSBSMPzjkZJDUNBUsNQkNQwFCQ1DAVJDUNBUsNQkNQwFCQ1DAVJDUNBUsOnOY/Cwal+Zc89t8gDkYbnnoKkhqEgqWEoSGoYCpIahoKkhqEgqWEoSGosOBSSnN61ijv079kkH5lVc06SZ2bUfHToEUtaVAu+eamqHgI2ASSZYPqx7TcNKP1mVV280O1IWlqjOnw4D/ifqvrRiNYnaZmMKhQuBW6YY9k7ktyT5N+TvGWuFdg2ThoPmX4C+xArSNYCjwNvqao9s5YdAxysqueTXAT8Q1Wd9nLrPCbH1Vk5b6hxSZrbd+t2nq2nM2jZKPYULgTumh0IAFX1bFU9301vA9YkOX4E25S0SEYRCpcxx6FDktemax+V5Mxuez8ZwTYlLZKh/nS66x/528AHZ8yb2TLuPcCHkkwCLwCX1rDHK5IW1dDnFBaD5xSkxbXY5xQkrSKGgqSGoSCpYShIahgKkhqGgqSGoSCpYShIahgKkhqGgqSGoSCpYShIahgKkhqGgqSGoSCpYShIahgKkhqGgqRGr1BIcm2SvUnunzHvuCTbkzzcvR47x2cv72oeTnL5qAYuaXH03VO4Drhg1ryrgNu7Pg63d+8bSY4DrgbOAs4Erp4rPCSNh16hUFV3AE/Pmn0JcH03fT3wrgEf/R1ge1U9XVU/BbZzeLhIGiPDnFPYWFVPdNNPAhsH1JwMPDbj/a5unqQxNZITjV0vh6GeFW8vSWk8DBMKe5KcCNC97h1Qsxs4dcb7U7p5h6mqrVW1uao2r2HdEMOSNIxhQuFm4NDVhMuBrw+ouRU4P8mx3QnG87t5ksZU30uSNwDfBk5PsivJFcAngN9O8jDwzu49STYn+QJAVT0N/A1wZ/fv4908SWPKtnHSLyDbxknqzVCQ1DAUJDUMBUkNQ0FS48jlHoBWlvfVg6xlarmHsSrtZ4Ib8+blHoahoPlZyxRfzluWexir0vvrgeUeAuDhg6RZDAVJDUNBUsNQkNTwRKPGWo7s/59oTU4u4kh+cbinIKlhKEhqGAqSGoaCpIahIKlhKEhqGAqSGi8bCnP0kfy7JA8muTfJTUleNcdnH01yX5K7k+wY4bglLZI+ewrXcXirt+3AW6vqV4D/Bv7yJT5/blVtqqrNCxuipKX0sqEwqI9kVd1WVYduH/sO001eJK0Co7jN+Y+BG+dYVsBtSQr4XFVtnWslSbYAWwCOYv0IhqVxdcRb+z9IZOqVR/WurSMGPrF8oEwe7FW3Zs8zvdc5+cijvWvH2VChkOSvgUngK3OUnF1Vu5OcAGxP8mC353GYLjC2wnTfh2HGJWnhFnz1IckHgIuBP6g5OspU1e7udS9wE3DmQrcnaWksKBSSXAD8BfC7VfWzOWo2JDn60DTTfSTvH1QraXz0uSQ5qI/kNcDRTB8S3J3ks13tSUm2dR/dCHwryT3A94BvVNUti/ItJI3My55TqKrLBsz+4hy1jwMXddOPAG8banSSlpx3NEpqGAqSGoaCpIahIKlhKEhq+DRnjcTE8a/uXbvvhA29a194zZretVPr5nGb81S/m2ZfcUz/7a9/4cXetZNPPNm7dqm5pyCpYShIahgKkhqGgqSGoSCpYShIahgKkhqGgqSGoSCp4R2NGo30//0yta5/7YEN/e9SnFw/nwe39qtb87OJ3uvkqHX9a8eYewqSGoaCpMZC28Z9LMnu7vmMdye5aI7PXpDkoSQ7k1w1yoFLWhwLbRsH8KmuHdymqto2e2GSCeDTwIXAGcBlSc4YZrCSFt+C2sb1dCaws6oeqar9wNeASxawHklLaJhzCld2XaevTXLsgOUnA4/NeL+rmzdQki1JdiTZcYB9QwxL0jAWGgqfAd4IbAKeAD457ECqamtVba6qzWtYHZd2pJVoQaFQVXuqaqqqDgKfZ3A7uN3AqTPen9LNkzTGFto27sQZb9/N4HZwdwKnJXlDkrXApcDNC9mepKXzsnc0dm3jzgGOT7ILuBo4J8kmplvNPwp8sKs9CfhCVV1UVZNJrgRuBSaAa6vqgcX4EpJGZ9HaxnXvtwGHXa7UKrSv/8nhiX0He9dmaiGD6bHefs9t5YjJnoVAPff8AkczXryjUVLDUJDUMBQkNQwFSQ1DQVLDUJDUMBQkNQwFSQ1DQVLDUJDU8GnOGompZ5/tXbtuT//bgQ8c/cr+g0j/Jy+n553W87klu/bt7107ztxTkNQwFCQ1DAVJDUNBUsNQkNTw6oPmZT8TvH/IB2gdsWdP/+0d6P8Q38lXzON3XM9np6x9vv9TXiZe/En/zdfhD6XZzzz6Vi4iQ0HzcmPePPQ6Jjae3rv2+Tf1vyT54rGjvyS54ckDvde57qmHetcePPBc79ql1ucZjdcCFwN7q+qt3bwbgUM/2VcB/1tVmwZ89lHgOWAKmKyqzSMZtaRF02dP4TrgGuBLh2ZU1fsOTSf5JPDMS3z+3Kr68UIHKGlp9Xlw6x1JXj9oWZIA7wV+a8TjkrRMhj2n8BvAnqp6eI7lBdyWpIDPVdXWuVaUZAuwBeAo1g85LI2zqQf6H3tvmOh/DmNi39G9aw8emV51R/7fZO91Tv+OXPmGDYXLgBteYvnZVbU7yQnA9iQPdg1rD9MFxlaAY3Jc/+dqSxqpBd+nkORI4PeAG+eqqard3ete4CYGt5eTNEaGuXnpncCDVbVr0MIkG5IcfWgaOJ/B7eUkjZGXDYWubdy3gdOT7EpyRbfoUmYdOiQ5KcmhjlAbgW8luQf4HvCNqrpldEOXtBgW2jaOqvrAgHk/bxtXVY8AbxtyfJKWmH/7IKlhKEhqGAqSGoaCpIahIKnhn05rrB2898Heteufe13v2lp/VL/CeTyh+eALL/auHWfuKUhqGAqSGoaCpIahIKlhKEhqGAqSGoaCpIahIKlhKEhqGAqSGqkav2ekJnkK+NGs2ccDq7F/xGr9XrB6v9tq+F6vq6rXDFowlqEwSJIdq7HD1Gr9XrB6v9tq/V6HePggqWEoSGqspFCYs7vUCrdavxes3u+2Wr8XsILOKUhaGitpT0HSEjAUJDVWRCgkuSDJQ0l2JrlqucczKkkeTXJfkruT7Fju8QwjybVJ9ia5f8a845JsT/Jw93rsco5xIeb4Xh9Lsrv7ud2d5KLlHOOojX0oJJkAPg1cCJwBXJbkjOUd1UidW1WbVsF17+uAC2bNuwq4vapOA27v3q8013H49wL4VPdz21RV2wYsX7HGPhSY7lS9s6oeqar9wNeAS5Z5TJqlqu4Anp41+xLg+m76euBdSzmmUZjje61qKyEUTgYem/F+VzdvNSjgtiTfT7JluQezCDZW1RPd9JNMNx1eLa5Mcm93eLHiDoteykoIhdXs7Kp6O9OHRh9O8pvLPaDFUtPXvlfL9e/PAG8ENgFPAJ9c1tGM2EoIhd3AqTPen9LNW/Gqanf3uhe4ielDpdVkT5ITAbrXvcs8npGoqj1VNVVVB4HPs8p+bishFO4ETkvyhiRrgUuBm5d5TENLsiHJ0YemgfOB+1/6UyvOzcDl3fTlwNeXcSwjcyjoOu9mlf3cxr5DVFVNJrkSuBWYAK6tqgeWeVijsBG4KQlM/xy+WlW3LO+QFi7JDcA5wPFJdgFXA58A/inJFUz/Kfx7l2+ECzPH9zonySamD4ceBT64XONbDN7mLKmxEg4fJC0hQ0FSw1CQ1DAUJDUMBUkNQ0FSw1CQ1Ph/eh5guTYpnVoAAAAASUVORK5CYII="
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "TODO - 20210508:\n",
        "- [X] troncamento rumore a 1e-6 tramite half gaussian\n",
        "\n",
        "TODO - 20210605:\n",
        "- [X] modificare RPNNet in modo che dia in output anche la backbone - Lorenzo\n",
        "- [X] scrivere bene training loop: salvare le loss in un df su disco + salvare pesi modello ad ogni giro (Lorenzo - finire di debuggare)\n",
        "\n",
        "TODO - 20210620:\n",
        "- [x] implementare mAP in una funzione che prende come parametro un modello o i suoi pesi\n",
        "- [x] implementare resNet50\n",
        "- [x] implementare predicted rois - Lorenzo\n",
        "- [X] implementare plot loss training - Lorenzo\n",
        "- [X] finire classe datasetv2 - Alice\n",
        "- [X] check se su colab le performance sono migliori - Lorenzo\n",
        "\n",
        "TODO - 20210627\n",
        "- [X] split dataset su combinazioni classi - Alice\n",
        "- [x] provare campionamento random patch ed osservare le due distribuzioni - Alice\n",
        "\n",
        "TODO - 20210703\n",
        "- [x] sistemare salvataggio loss training loop - Lorenzo\n",
        "- [x] Riscalare immagini tra 0-255 - Alice\n",
        "- [x] capire se passare tre immagini diverse come input\n",
        "- [x] usare media vgg16 per zero-centering - Alice\n",
        "\n",
        "TODO - 20210705\n",
        "- [x] sistemare nomi funzioni dataset per trasformazione rgb\n",
        "\n",
        "TODO - 20210711\n",
        "- [x] rifattorizzare classe dataset spostando nel costruttore i metodi che calcolano i suoi attributi - Lorenzo\n",
        "- [x] chek valori pixel in input per resnet\n",
        "- [x] fare funzione per plottare le predictions\n",
        "- [ ] trainare tutto su colab\n",
        "\n",
        "TODO - 20210714\n",
        "- [x] ragionare su come scalare le immagini fra 0 e 1, attualmente hanno tanti valori schiacciati a 0 e il massimo su tutto il train a a 0.4\n",
        "\n",
        "TODO - 20210717\n",
        "- [ ] Ablation study: provare a rimuovere stage4 nella resnet - se c'è tempo\n",
        "- [x] Provare con nostra pixel_mean e con vgg16 pixel_mean -> per il momento abbiamo scartato la prima opzione\n",
        "- [ ] Fare qualche analisi di distribuzione delle classi/dim box del dataset - Alice\n",
        "- [x] Aggiungere normalizzazione dopo zero centering per resNet50, sulla base del range globale dell'immagine di training\n",
        "- [ ] Provare pulizia dataset originale sulla base del rumore/flusso - Alice\n",
        "- [ ] Cambiare nomi di tutto - alla fine\n",
        "- [x] implementare zero-centering su volare medio RGB delle nostre patch\n",
        "- [x] Funzione che trova l'ultimo checkpoint in colab prima del load_weights - Lorenzo\n",
        "\n",
        "TODO - 20210801\n",
        "- [x] Debuggare training baseline 8 e 16 - L\n",
        "- [ ] Finire prove pulizia dataseet noise variando k - A"
      ],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "1.  Summary\n",
        "    - riassunto progetto\n",
        "2.  Background\n",
        "    - SoTA + teoria di base\n",
        "3.  System Description\n",
        "    - descrizione dei nostri modelli e dei loro componenti (moduli)\n",
        "4.  Experimental setup and results\n",
        "    - dataset pre processing\n",
        "    - training environment\n",
        "    - metrics\n",
        "    - results\n",
        "5.  Results and error analysis\n",
        "6.  Discussion"
      ],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Plotting"
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": 71,
      "source": [
        "loss_history = np.load(f\"./model/{backbone}/loss_history.npy\")\n",
        "print(loss_history.shape)"
      ],
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(422, 5)\n"
          ]
        }
      ],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "source": [
        "loss_history"
      ],
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "execution_count": 72,
      "source": [
        "plot_loss(loss_history)"
      ],
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1080x360 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA2cAAAE/CAYAAADCCbvWAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABjZElEQVR4nO3deZgcVbk/8O87WzJJZk0mIWSHBELYdUR2RETDzkXlBoELiqBeFC+I/kC8kU3ElUXAKyKbqIEbUSKg7DsECIQtJJAFsi+TZZLMZCbJzJzfH2+dW6erq3p6Znqmu2q+n+eZp7urt9PVPdX17ffUOWKMAREREREREeVXUb4bQERERERERAxnREREREREBYHhjIiIiIiIqAAwnBERERERERUAhjMiIiIiIqICwHBGRERERERUABjOiHqJiBgRmRhx3bMi8vW+bhMRERERFS6GM+p3RKTJ+esQkRbn8pkR9/mMiKzo67YSEREVAhE5V0RezHc7iJKuJN8NIOprxpgh9ryIfAzg68aYJ/PXIiIiIiIiVs6I/o+IDBCRG0Vklfd3o7dsMIB/AtjVqbDtKiIHicgrItIoIqtF5BYRKevG8xaJyI9EZKmIrBORe0WkyrtuoIjcJyIbvOd5XURGeNedKyJLRGSriHwUVfUjIiIionhgOCPyXQHgYAAHANgfwEEAfmSMaQZwHIBVxpgh3t8qAO0ALgYwDMAhAI4B8J/deN5zvb+jAewGYAiAW7zrzgFQBWAMgKEAvgmgxQuMNwM4zhhTAeBQAG9147mJiIhSiMgYEXlQRBq8HwdvCbnNTSKyXES2iMgbInKEc91BIjLHu26tiPzaWx75g2OGtnxVROZ7P0QuEZFvBK4/RUTe8p5rsYhM9ZbXishd3o+tm0Tk7zlZOUS9jOGMyHcmgKuNMeuMMQ0ArgJwdtSNjTFvGGNmG2PajDEfA/gdgKO6+by/NsYsMcY0AbgcwDQRKQGwExrKJhpj2r3n3OLdrwPAPiJSboxZbYyZ143nJiIi+j8iUgzgYQBLAYwHMArAjJCbvg79MbMWwJ8B/K+IDPSuuwnATcaYSgC7A3jAWx76g2MnTVoH4EQAlQC+CuAGEfmE19aDANwL4PsAqgEcCeBj735/BDAIwN4AhgO4odMXT1QAGM6IfLtCv4yspd6yUCKyh4g8LCJrRGQLgOugVbRcPG8JgBHQL5fHAMzwfv37uYiUetW8f4d+sa0WkUdEZHI3npuIiMh1EPR76fvGmGZjTKsxJm0gEGPMfcaYDd4PlL8CMADAnt7VOwFMFJFhxpgmY8xsZ3nUD46hjDGPGGMWG/UcgMcB2CrdeQDuNMY8YYzpMMasNMYsEJGR0B4v3zTGbDLG7PTuS1TwGM6IfKsAjHMuj/WWAYAJuf1vASwAMMn7dfCHACRHz9sGYK33hXKVMWYKtOviiQD+AwCMMY8ZY44FMNJrx++78dxERESuMQCWGmPaMt1IRC71uhtuFpFGaEXM/kB5HoA9ACzwui6e6C0P/cGxk+c5TkRmi8hG73mOd55nDIDFEa9hozFmU6evlqjAMJwR+f4C4EciUiciwwBMB3Cfd91aAEPtQB2eCgBbADR5Vatv9eB5LxaRCSIyBFqBu98Y0yYiR4vIvl43ky3QXx07RGSE189+MIDtAJqg3RyJiIh6YjmAsV7X+lDe8WU/AHA6gBpjTDWAzfB+oDTGLDTGnAHtTvgzADNFZHCmHxwjnmcAgL8C+CWAEd7zPAr/h9Dl0G6TYa+hVkSqs33RRIWC4YzIdy2AOQDeAfAugDe9ZTDGLICGqCXeQcy7ArgUwFcAbIVWre7v5vPeCf018XkAHwFoBfAd77pdAMyEBrP5AJ7zblsE4BJo1W0j9Fi37oZDIiIi6zUAqwFcLyKDvUE8DgvcpgLaw6MBQImITIceEwYAEJGzRKTOGNMBoNFb3BH1g2OGtpRBu0s2AGgTkeMAfN65/g8Avioix3gjH48SkcnGmNXQUZZvE5EaESkVkSO7tzqI+pYYE9Zbi4iIiIj6IxEZCx0R+Ahot/4/Q3+w/Lox5nAvXP0ewJcANEMH2/hP7/onReQ+aIgaBD2O+gpjzN9F5AwAVwIYDe3xcT+ASzJ1oRSRC6E9WQYA+AeAUgCLjDE/8q7/N+gAXhOgvVwuNMY8JiK1XrumQkPeM8aY03K0ioh6DcMZERERERFRAWC3RiIiIiIiogIQebAnEREREVFvE5GmiKuOM8a80KeNIcozdmskIiIiIiIqAOzWSEREREREVAD6tFvjsGHDzPjx4/vyKYmIKA/eeOON9caYuny3Iy74/UhE1H9k+o7s03A2fvx4zJkzpy+fkoiI8kBElua7DXHC70ciov4j03ckuzUSEREREREVAIYzIiIiIiKiAsBwRkREREREVAAYzoiIiIiIiAoAwxkREREREVEBYDgjIiIiIiIqAAxnREREREREBYDhjIiIKISITBWRD0RkkYhcFnL9OBF5SkTeEZFnRWS0c905IrLQ+zunb1tORERxxXBGREQUICLFAG4FcByAKQDOEJEpgZv9EsC9xpj9AFwN4KfefWsB/BjApwEcBODHIlLTV20nIqL4YjijZGluBl56Kd+tIKL4OwjAImPMEmPMDgAzAJwSuM0UAE97559xrv8CgCeMMRuNMZsAPAFgah+0uXPGAI8/rqdERFRwGM4oWc46Czj8cGDt2ny3hIjibRSA5c7lFd4y19sATvPO/xuAChEZmuV98+Pvfwe+8AXg5pvz3RIiIgrBcEbJ8uabetramt92EFF/cCmAo0RkLoCjAKwE0J7tnUXkAhGZIyJzGhoaequNqTZt0tO5c/vm+YiIqEsYzihZOjr0tIgfbSLqkZUAxjiXR3vL/o8xZpUx5jRjzIEArvCWNWZzX++2txtj6o0x9XV1dTlufoTaWj3duLFvno+IiLqEe7CULAxnRJQbrwOYJCITRKQMwDQAs9wbiMgwEbEbm8sB3OmdfwzA50WkxhsI5PPesvwbMkRPGc6IiAoS92ApWdq9HkUMZ0TUA8aYNgDfhoaq+QAeMMbME5GrReRk72afAfCBiHwIYASAn3j33QjgGmjAex3A1d6y/LM/YMUxnH34IQcyIaLEK8l3A4hyipUzIsoRY8yjAB4NLJvunJ8JYGbEfe+EX0krHHENZ48+CpxwAnD//cDpp+e7NUREvYZ7sJQs9ldVkfy2g4ioENneBXELZy+/rKfz5+e3HUREvYzhjJLF/irMri9EROlsONu5M7/t6KrNm/W0qiq/7SAi6mUMZ5QsDGdERNHsNjJubDgbNCi/7SAi6mUMZ5QsDGdERNHanWnY2try146usuFs+/b8tiNbZ58N1NfnuxVEFEMcEISSheGMiCiaWznbtAnoq/nVesqGs5aW/LYjW/fdl+8WEFFMsXJGyRLXLjtERH3BrZzt2JG/dnRVY6OebtuW12YQEfU2hjNKFlbOiIiixbVb49q1ehqXyhkRUTcxnFGyMJwREUVzexfEJZy1twPr1ul5Vs6IKOEYzihZGM6IiKK5lTP3fCHbssXftrNyRkQJx3BGyWJDGcMZEVG6OHZrdANZ3Cpn/C4ioi5iOKNkYeWMiChaHLs1uoEsbpWzOA26QkQFgeGMkoWVMyIi4O23gQ8/TF/Oylnfam3NdwuIKGY4zxkREVHSHHCAngZ/qIpj5cwNZ71dOXv6aaCoCPjMZ3LzeK2tQFVVbh6LiPoFhjNKJlbOiIjSxXFAEFstGziw98PZ9OlAcTHw3HO5eTxWzoioixjOKJkYzoiI0sW5W+PQob3frXHbNqAkh7tGDGdE1EU85oySieGMiPort+tipuuam+MRHmwgGzq09ytnLS25fY44rF8iKigMZ5RMDGdE1F81NUVf51bOjjsOGD6899vTU31dOctloGI4I6IuYjijZGI4I6L+avPm6OuCx5lt3dq7bcmF7lbOXnoJWLq0a8+Vi8qZW51kOCOiLmI4IyIiSpLGxtTz11/vh7KwLo+FPhdXdytnhx8OjB/f9efqaaDaudM/z3BGRF3EcEbJxMoZEfVXbuXsqquAyy8HZs7Uy2EjNHa1utTbtm8HzjsPWLZML7vhrK0tNfx057HPPTd8Djj7XD2tnLlhl+GMiLqI4YySieGMiPorWzkrLvZHHvz4Yz0Nq5wtXtwXrcrevHnAnXcCjzyil221rKZGT3sSnj74ALjnHmCffdKv27lTw2tra9e/Q77/feD88/X+bjjbvr37bY1y1VXAV7+a+8clooLAofQpmRjOiKi/spWzigpg5Eg9v2qVnoZVzgotnG3YoKfLl+tpSwtQXg4MHqyXt20DKiszP0bUHG422O3cCaxcCYwapZeN8ddRR4deX1aWfZt/+Us93Xdf4Mtf9pe3tgLr1wPDhmX/WJ258ko9veuu3D0mERWMrCtnIlIsInNF5GHv8gQReVVEFonI/SLSha0YUS9jOCOi/soNZ+Xlen71aj0NCy0ffdQ37crW+vV6unw5sHYtsGABMGiQH8iyGcQkqrrW3Jz+PADwi1+kHp8W1h2xvR14443wx504UU8XLEitnL36KlBXp9U6IqIsdKVb43cBzHcu/wzADcaYiQA2ATgvlw0j6hGGMyLqr2y3xooK//gstyoUZCtVhcINZ6NHAw8/rCGzqkqXuwOeRMkmnLlB9cEHO7//9OlAfT3w3nvp19mulxs3poazJUv09JprOm8zERGyDGciMhrACQDu8C4LgM8C8I4wxj0ATu2F9hF1D8MZEfVXtnJWWuqHs0yVs40b+6Zd2XK7Nba16Xk3nGWaKsByw5X7feCO9mgfG9B15QqrnD39tJ66Fbfg4wbDmW3r4sX6fHfdBbzwgh43VuijZBJRXmR7zNmNAH4AoMK7PBRAozHGbtlWABiV26YRERFRl9nKkjs4RaZjzgotnNnws2KFv2z7dqC6Ws9nUzlzw1Vrq9+9062ctbVpcLvoIh2ExOWGu5tuAnbd1Q9gYeHQ3n7TptTQ5Qa5ZcuAr33Nv7zPPsAXv9j5a3G5QbO9XQd9IaJE6bRyJiInAlhnjInoaN3p/S8QkTkiMqehoaE7D0HUdaycEVF/ZYOJHdgiuCxo06a+aVe2bKBxQ05DQ+bK2bRp/kAZQGq4cm8f7NbY2Ajcckv6OnDD3X/9F3D66X44C1bO2tv9URmDlTP3tsEQPHgwcPHF2l3ykEN0JMnOuJW/rsz5RkSxkU3l7DAAJ4vI8QAGAqgEcBOAahEp8apnowGsDLuzMeZ2ALcDQH19PfeYqW8wnBFRf2WrY+3tqXOCdXTEo3IWdgxcS0vmY87uv19PL7xQB+Bww1ljI7DLLno+2K2xqSm8DS0tGgjr6vxlNtjZH5obGoDa2tTnClbO3NDnVgIBDXQPPgisWaP3eeklYM89gS1bgAED9C/Ife3btulxhUSUKJ1WzowxlxtjRhtjxgOYBuBpY8yZAJ4B8CXvZucAeKjXWknUVQxnRNRf2QDmVs4AHUlw3br046sKsXK2667pyysqAJHMx5ztsgvw/PPhlbOlS4FFi/zlbW2plTTXs88CI0YA777rL7Ohcf16HRRk+HDgd7/zA9/QoRqeogYjsVMDWNu2acCzYW7dOj09/HDgxz8Ofwz3vWLljCiRejIJ9f8DcImILIIeg/aH3DSJKAcYzoiov3LDmVvFOf54rTC583eVl2sXvp5M7NwVxgD33Zd5MIz164EDDkhfXlSkw+kHw5k70XNHhx7b5XZLtNWm8eOBO+7wl7e3R1fO3n9f2+oOnW/b3NAAXHutnl+40A9Jo0bpfaIO4Vi2LPVyQ0PqerfhbMkSf5THoGDlrD974on0wEuUAF0KZ8aYZ40xJ3rnlxhjDjLGTDTGfNkYs72z+xP1GYYzIuqvoro12sqPG86GD9fTvqqevfMOcPbZwD//GX2bDRuAvfZK7db3rW/paVVVerfGYDfI5ubU0PNf/xVebcvUrXHNGj11K23Wvff63SiHDPFD0ujRerp2bertBw3SUxskbrxRT5cuTb3d2rV+NS9q0BN3uVv1e+ghnXKgPzntNOA3v8l3K4hyrieVM6LCxXBGRP1VVLdGG1jc0DNihJ721XFnW7akngZt26btHDbMDzuvvALcdpuer65OD1qdhbMFC4Bf/jL9uboTzuzoiIcdpuuxuTm1cubet7ZWT3fZRat+NpztsYeeBitp69b5ry2q62ZYt8aODuDUU4GTTgJWrgQ+9zngK18JH/wlKYzR9y6qWypRjDGcESXdRx/pnDoMrET9g1s5c7sP2uXuMWe2ctZX4cwGiqidaju64bBhwJgxet6GHCC7ytlNNwFnnaXnTzhBX2NYhaW9PbodUeHMrsOvf12DohsEbTizlTMbLisr9bY2jNXV6bFzwcpZVDhzt93uoCJHHw3MnKnh1Zo9G3jqKeAvf0kfgMT19NNanYxr10jbbTVsPjqimGM4o2RiEPH94x86xHShjchGRL3DVkyClTPLnRvLVs76qltjZ+HMBi03nA0d6l9fVdV55cytSN11F/DDH6bep8jb9clUObMhMRjOhg3T05NO0qHw3cqZba8NXfZyRYWGMxuWKiu1q2OmcGYD6Pe+p+01RkPYD3+Yep8vf1m38Za7nV+8OPy1AcCll2pFcc6c6NsUMoYzSjCGM0omhjOfXRdcJ0T9Q9QxZ5YbzuxQ8X0VzmyVKapiY0PR0KE6rLwNNlZ1deeVM1d5efpEzTa8ZgpnVjAI/uMfwHPPafsGD9b729ey//66Pl96SS/bytmAAUBNjf8YFRUazuwAIFZDgx+u7PP++td6+tZbGqYA4JJLUu/3/PP+eTeY2nDW1pZaXQP80TCDk2/Hwfr1wNy5en47hzugLlq4EFi1Kt+tyIjhjJKJQcTHcEbUv0SN1mgVOV/9du6wvure1pVujZdcoqHEDVfZVM5cYeHMytStMcrEicCRR+r5IUNSK2cVFXrsV1ubXraVs9bW1IBpwxmgbRsxApg8Wdvz0Uf+fbZv166LgM6HZoPbBRektskNXgsW6PtbWuqHs+nTgUMP1XVp2eMObciJk6uvBo45Rs+zckZdtccewH775bsVGTGcUTIxiPgYzoj6l6gBQSw3rFRW6mlfDaXflW6N5eXAbrulXl9RAWzdmro9yxTOioujw1lY5Wz48OjbAxrIrGC3xkGDdARBy1bOtm1LrZwNGuSHs2HDNJBdfbVedudV27zZf39eflmrm8XF/oTarkMO0dMPPtDq3fjxfjibPVtP3fVkj4t7883o19pTf/sb8K9/5f5x3dEwGc7IevJJ4IEHMt/GdiXOtM0oAAxnlEwMIj6GM6L+pbNujW7lLF/hrLNujW6YcVVUaKhyu7MFuwcGFUXs6gTD2fTpuuM/cGD0Y7kjXQbDWXk58NnP+uvUhrPmZr9yNmSItmfwYL1cV6f3mzJFL7/6qv/4mzf74WPDBq2c1dT49wV0fQDAUUfp6YIFepvdd/fDmV1X7nqwA57Mm+d/XnLttNOA447L/eNu3eqfZ7dGsn796+jJ260XXtBT93+oADGcESUdwxlR/9KVyll5OVBSUjiVs/XrNVyUlIRfb8OI3UHfvBmYNSs6gAGZK2duO2w1y4Yqyw4CAugoi5YNZ3bdDRqkc8iddJLezq2cTZyo523Yss81cqSeTpyor8ENZ3vs4Q9IsnGjVs6C6+aww1JPd+7U0S3r6vxukDbAuIF47VoNiq2t6QOTFDo3nLFyRlZjY/T8gJY9HnTChN5uTY8wnFEyMYj47LpI8pw3ROSLGkrfcsNKaakGtEIJZxs2pIahoGA4+/OfNaBdc030fYLhzAai9vbUypkNTAcemHr7ffcNf9zgMWe24nb11cCdd/rTFOyzD3DEEXreHo8WDGcDBmi1K8hWv2zlzJ1WANAJsW++GTj2WH9ZTY2GxB07NITZ12jXWXOzLrPHs82fH/76ClVUOGtvB8aO1c8EoHPpRc0XR8mzeXPnAxvZHzv6anvXTQxnlEwMZz5Wzoj6l86G0nerTDac9dWAINmM1tiVcLZqlVapbPgJ477efff1RzcMdmu0XZ2CgwVEhTN3tMZBg/yq2m67AeeeqyHp2Wd1zrH6+tT7lpfrqXv82F57Rb+Gbdv0tQbDWV0d8J3vaLizUw7U1mo4275dH/+DD3S5fa32mC3bFTLO4czt1tjUpBN929dbXZ06EAslW2Ojfh4yVVNXrtTTzkZpzTOGM0omBhGf3VHjOiHqH4LHnNkgYBVy5WztWn94/zA2nNmdq+ZmDUnuQB1B7us9/HA/EEV1awwOQrL33uGPa7s1Njf79w066igdYbKsTCtp++yjy21F01bOAOBTn4p+DYD+6u8ei2e7MgbvbytnwXUcDGdTpui6zhTOFi8GHnkkc7vCuMex5fqYNnfH2t0Rt58t+1nmd17/EpwjMIwNZ10dpbWPMZxRMnGj7GPljKh/CQ6lHwwOblgpKem7cPanP+nxYUD4zpExwJIl6eHIFayc2XAWdoD/1Kl66r7eoiL/mK1Zs3TOMrt+7Okxx+h9zjxTuypOnhzelsGDtc3r1vntymTFCn/oersj6YazSy8FPvc54CtfSb2fHYSktdWvnO3YoW13HXqonhqj4SxYQbDrzA4GsssuwKRJus6jTJkCnHhi568tyH1/t2zJ7j4LFwLXXdf5d1VUt8ZgOKPc+s1vgNdey3cr0q1fr/879jMX1bVx61b9LA4cqLe1P1zfc49umwoIwxklE4OIj+GMqH8JDggSDGdh3Rr7Yof2rLP8QSreew/49rdTr1+7Vneawo69sqLCWbBydt99wD//qefdcFZc7IezF1/U01131SBnK0/DhmlV7b77dL3YueCCbCBctCh8ePug0lL/uW04c+83cCDwxBO6o7h6tb/cThgN+JWz0tL0Y+nssXINDRrOgoKVsxEj9PkzjXZpK3xd7QbmBqjOBmn43veAhx7SEHjFFZnb09aW+lntajgzRueJe/rpzG1KmgULgC9/OfvRLadP1+64rosuAj796dy3rad++EPgV7/yL0d93mzVbM899XPQ0qLzB557rm6bCgjDGSUTg4iP4Yyof7HhzBjdue5Jt8bp0/VX6d5w662pFRY7+EV3wlmwcuZedsNoUVF6qLn5Zg1ybhXLFdVl0QbCxYuj7xvFVpOi7ue+Z244Cx5z5po6Vd+r667LHM7WrNHj4+rqtKtlMAwZk/590dl0BUFuOLOVjLBjII3RIdBPPdVvX6bjH4Mh0Q0b9jMcrBi6r2XdOuD3v+98PqxcOfRQ/R/Kt/POA2bOTB0NNJNrrkmt4HYW6m6/Pbr7by6EfSat4PsdVTlbtkxP99xTT5uagGee0fNVVcD++wP339/ztuYAwxlR0jGcEfUv7jE+ra09q5w98wzw1FO5b6PldqmzI6l1JZw1NfnhzA0k7nDzwcqZiL8ODj+887m4guHWsgGwuTm7ypnLHntmR3TM9JxuOLODfoQpKQF+8QsdJry0NP16u87WrtXHKS3V59+wwR9F0j7O8cenfme8/XZ4aDIG+Pjj6OcCtJLR1KQ77+6okkBq2LLvma18LF6c2obly9N3vHfs8LunRVXOtm3Tib4B//g6+0NA0LZtXQ+imbzySupIops2pY8guX596voKam7WamhP2HAV9rnIhts1NayK+o1vAO+/n1rxzaV99kmf4sJy5x4Ewitns2cDX/iCnrfhrLnZf12bNwPvvANMm6afpw0bUh9nyxZ/DsY+wHBGyeEOFc8g4mM4I+oWEZkqIh+IyCIRuSzk+rEi8oyIzBWRd0TkeG/5eBFpEZG3vL//6dOGu+Fs+/bOK2eDBkWHMzvsem9xd5IXL9bQNH589O3DKmdDhuiO/bPP+l0Tg4EseN4GgbAKU1BU5cytznW1cvanP2m3yqjRBEtL/QDphrNsJ3XurHJmw+SIEfrd4O54dnQA//pXaiA47TQ9Fi/o17/WMLhgQeryYOXsllv0Ns89l/r5dJ/XdnndvFlvO3GiP6nwxo06TP43vpHehuA8bi0tqd93f/6zHsc4a5YfzuwPAUGHHqrrpCfa2vSzGPY/VVub/p7X1UWPCAoABxwQHeKD5s4NDxG2e2p3R2V1A+ULL6RX4Ow6e/PN7j0+oOvt8cfD91Xef19HKw16/XUNUi4bqtatA956S88/9JB/vVs5CwvFr7yiQdD9weVTn8o8UFGOMZxRcrgbfAYRH8MZUZeJSDGAWwEcB2AKgDNEZErgZj8C8IAx5kAA0wDc5ly32BhzgPf3zT5ptOX+UBVWOetKt8a+CmdLlwJ33KEDUGQKTKWl+kt5sFsjABxyCHDhhXre7WIV7NYI+Osgm3AWVTlzq1hdDWdVVemjLbpE/IpAVZW2d9q0zNMMuNzX9dOfAgcfnHrMmd2Ztjv9tlrkzosXnJx69uzUChsA/PWvqbfdsgW48cbUYNfY6A9vD6RW2tzb2fZt3uyHqGuu0a6Q9nZPPJH+WsPCmdvV7e239XTmTP9xly0LnwPQ3rYn/vEPnUMumy6E9ns500TgUUEy7LE+8QngyCPTr7PrKDg4y7x5GsRdYXOiuvc7/nj9PLnLxozRUzvgTXc89JBWt37+8+xu39amr/Vvf0tdbqurBx7oH4f54osachcu9P+HmprCB6v5+GP9/Gzf7gfdDz/U055WMLPEcEbJwcpZOIYzou44CMAiY8wSY8wOADMAnBK4jQFQ6Z2vAhDy024eBLs1BsOFG1Y6G60x6tflXLE7nnfeqaFhxozO71NRER7OAOCcc3RnfuxYf1lPK2cDBwKVlcBtt6UuP+AA/3xXuzVmw7bRji5nJ1fOhvu6pkxJXWduOLOnNpy5O59hgeGFF1Iv253bdev0O3i//YCLL9aBFqxNm1Ifyx26P6zK09iYWlH94IPMg4rYIOaGM7dCZCtyf/yjjjgIaFszBaJgQDEGuOoqHcimM7aS4wbSKG44yPY7eu5c4Prrox8rbGqEqHC2zz7p1Vg32NrzYZN5266igB/ae1o5A3SAj2zWxcqV4XOa2c+KrbQ1Nekok5/7nFZj7bGibrdGlw1iQGrFDfAHEeplDGeUHKychWM4I+qOUQCWO5dXeMtcVwI4S0RWAHgUwHec6yZ43R2fE5EMMyT3Andb2NHR88qZO+x0rtmd8Dff1EmYsxlUwA0a9pgzl3u8GZA+lL57m2zCmYjunH7rW6nLBw70z3e1cpYNO6n1gAH6Zy9nw31dZWW6QxrWrdFWzuwIju7xVnZ0O1dw59vu3K5YoVUnG3hsdzJAw9GyZf4xP254CKtEbN6cWi3auDG965rL7qC7A4K4A824jzVqFHDGGXo+eNyZ+x0ZnOph/Xrgyiu1OtsZGwzdCuHbbwP//u/+Zfv/tGKFvyzTa3R94QvA5ZenB9tMx3vZKmE2P7S42wL7eQgLMe56tdUqO+hGd9gA2dGhz7d5M3DCCeldZq2w5yov98O4NXu2vv5PflIv23Bmf3iy3ajt/8krr/j3tT9G2NsEf5zoJQxnlBwMZ+EYzoh6yxkA7jbGjAZwPIA/ikgRgNUAxnrdHS8B8GcRqQzeWUQuEJE5IjKnIZfdZYKT/kZ1ywOyC2dA949VifKDH+iOlw0Fc+f6XZA6k6lyFsatFHancpZJfb2e9kblzLY7OOBBNsLC2datukO6bVt0t0b7fgC6s+8OIDFoUGpg27RJB+kAdLm742qrDxMn6nFmy5drpXHECODvf9dq3l//Gl4527xZg5N9rzZuTN/hdoV1awwbBfSyy7RdP/uZXg7u3LttCXbltaHTXb5zp36v/vSnesxY8LgutzL32GOpI0TaIOauTzcsbtyonyl3MJ72dn0+O+JlsAuhDWdFIbv2UZUza+1afU/mzs0+nAXbC4SHv61b9XH+53/0NQX3RVat0rkEb7nFX7Zpk/4Q8OijqRVqt1ttWOXzE5/Q7qvu4CFz5uipDVh2e3HaaRr87LyKu+yi3TNfflkvV1bqHGjDh/vvUzaV0xxgOKPkYDgLx3BG1B0rAYxxLo/2lrnOA/AAABhjXgEwEMAwY8x2Y8wGb/kbABYD2CP4BMaY240x9caY+rpcHmweDGfByplbBbPhbOvW9J2dHTv8naFcHHdm2/G5z+kO8vDhulO3bp3u/HQlnG3Zots0OyBIJmHdGrtyzFkmjzwC3HVX6qAduZKrcDZggK6zpUv1mBvAn7utulqvtzufbuVs4cLU0DlqlH87Y/SytWIF8PzzwLhxuoNs23/WWbqzu2OHdjU97TS9PH8+8KUv6fnS0tT1Z7s12sFdNm3KrnJmQ1Fzsx5LZW3cqBXZn/5UP4O77KJtC1YG3c9/8PPuBrl16/yJvuvrtRvee+/597fB0H284I8vs2fr/2lUOHvwQQ1MV17pLysp0bkB7bbCDWfNzf7lQYNSq1rG+OEqKpw9+6y+J0884X9G7GsFwrs12vbu3OmHsmA427xZ39sRI7TyvHZtehvOP1+7gL7+ur/MHdXSHcbfDd3u+p06VcPv0Ufr47vr9Y039HTcOD0Nbi9sOBs5Uj/T9jlO8XqxNzT4gThqlM8cYzij5AjukJBiOCPqjtcBTBKRCSJSBh3wY1bgNssAHAMAIrIXNJw1iEidN6AIRGQ3AJMALEFf6U44A4BJk1J3St0dIXena/781ONNslVSoqPhzZypl2tqdCfM7kjvv392j1NTozvwra26XeuscpapW2N3hxa3hg/XSWx7g+3G6HafzFawcmYrZSedpKd2nYloKJ49Wy+7lbP339cgs2iR7rCPGuV3w9u6VSss9fXAZz6jy194QQdosFMhDBmiEx9b48YB3/++tufss/V1PfigDtBgJ9cGtAK0bBlw0EF6OditMVgZCoazlStTnxdIHWmvtFTXh9ulEEjd2Q+GDPe6117zw4/bzdMGuLBujcEfPk4+WStFbohw/6cefVRPg8H8ttv8x3fD2Wmn6WTegP4PT5oEvPSSXt682f+RxQ1GbjXcBpj/9/+Az37WX24/D/Z+7v+aDSr2GK+SkvT1duut6UHX7X5pjN9OV1S11H0sNzBXVup7fkRID/I33kj9HwiGs1131fVswxmg24ypU9Mfa8kS/WEpbOTIHGI4o+TggCDhGM6IuswY0wbg2wAeAzAfOirjPBG5WkRO9m72PQDni8jbAP4C4FxjjAFwJIB3ROQtADMBfNMYk6FfVo61t6cedxXs1hgVznbu1Ll+LDecuTtFU6b4vzZ3tV0HH+xXbWpr9TnsTnK2XQNtqLPt60o4y3W3xt6Uy26NP/iBVq5sGHDX2ZFHauBoaUmtnC1YoDusu++u3RNHj9YRCO+4wz/G6Jvf1OvfeEPve8QR/g7umDH6WbniCr28zz7+sPt/+IPeF9CddTecPfaYflY+/3l9v4KVs2CVOditMUxwlEu3Cmi5O9xR3RoB7WoYFsjtbWw73McL+zFj/nz97A8bpkE1OFw9kPr/aK1Zo6cvvuh/r4eNYmnDk9td0w1P9nGA6IE87H03b9bPofs+vfaaBjP7WRg7Vtebu32ZOVN/kAm2f8cO4IYbgHffDa/KRVVL7fsyd65OfG1HTLVhPPhcgK77MWP8/6fgj1WVldot9YAD/MebPDk96NnpDi67LLUy2wsYzig52K0xHMMZUbcYYx41xuxhjNndGPMTb9l0Y8ws7/z7xpjDjDH7e0PmP+4t/6sxZm9v2SeMMf/o04Z3dKRWhDKFMztao+XupLk7qLno1tjenhqU7I6erUJkmmDZVVurv6rbNnXlmLPuDKWfL+6AIF3lvv8DBujOv1sRcXdQjzhCg/mrr6ZPwOwOdDJqlL6H55/vVzuqq/0JtQENescdp3NJ2ZE3r71Wq1u2W5mdJPv66/W4w4su8gM7oO+tiE4QXl2dXjkbORI49VQNjEB65SxMMNCNHp0ezjJNtLx0qVajAJ0qwB6XFLxNVDvCwtmwYRpMd91VA4INKcb41aiwY1HtjxwrVuh0DO+95x/76LIVV3tcIJD6Gt0KVlQ4a24Gnn5ah7evrPQ/N1VVGvRuvdWvcNn31/1RZ/16XW/ByuiDDwKXXKJV1zBRg8DYcPmLX+jpBRfo+vjSl/TykCEa/Oxnw7JtA3QbsHOnP2diZaV+9qdP10nS99hDu0iOGZO6/t1j36LanSMMZ5QcDGfhGM6I+pf29vSBHFzBypm78x88jsWyO0U92Y4Ew1ltrZ7acGYvd6amRncyw7pahWHlTE/d9eSet8eILVigO6PuZ8etZrpB7Q9/0NOaGuCrX/WX77GHVrwWLEgNbWGvYcAA4OGHgZtuSv+M7r+/BoDa2vQqyi676NxWNvy1tgK/+51WUoJsKAtO/Ox20bTcqlLYMWcTJ4bPM1dcrI9nu9kFR3oEwoPGpk0ahkeM0NdqP8+trenzyQXZquMrr+iE5mETZ9sBSuzomBMmRIczW/0K2rZNQwugr8u+T4ceqvOd3XCDvx5tANq6VcPO6afr56m2NnW9rV7tzwFnn9ceX2g/qzacBavp9n1ZsQI46ijguuvSu7mWlupgM6ef7rfXDWeA/v/b7o0VFfoYItqV8YMPtOILpP5gZMNZZWXPu0N3guGMkoPhLBzDGVH/0t6eunOeKZyJpFZL7BDoweHI7U5RtnOebd/uH0TvtiuscrZwoe4gZRuUbIizx6jlc0CQ3mR3OnNxzBmQup7ccDZ8uH4OVq/WHWJbIQJSA5nbjqef1tPqag0WM2cC997bteH+XcGd3Ysu0tOamvQqig1atj3bt/thJeiyy/Q0OBXEqFFanWpu1opVdXXqHHthx5yNG+evjy9+ETjvPL+NNTXA3XfrEPdu5SwsNJ1/vp5u2qTHdA0f7lfOWlr8Clrw/9a1775+GN22zR9h8eij/du4855VVGhXva1b9X98zz119MTOLF7sVwm3bfPbNGiQDoSyYYM/d5ydW7CpSSe2/t//1e1ITU1q5XLNGh08ZtdddVqDn/3M3xbU1mpot4E8WE13w5k7ImPQfvsB99/vvxfuerHs/0Bl2kC6PvfzPHmyDtBij8/sRQxnlBwMZ+EYzoj6l2DlLDhHVnBbYHeAqqt1x+mjj3SH+8kn/dvYnaLg0OctLakDH9jHP/bY1HmdjNEd5LDK2cKF2XdpBFK7SAGZd2KB8G6NViGHs550awyO1gikhjN3nZWU6M7zmjW6Q7yHM7DoGGfA0nPPBX77W+A//9NfZoPSF7+og3x0l/28fuc7WvWw1biwypnthmt3qv/+9+jH/frXdYLzSy9NXW6HVf/rX3XEzc2bNYDZxwwOjLNhg4YzW8kZN87/HFZX+493662p4SwsQEyfrpUiO1Lp8OH6//bPf+r7Ykct3HPP6NdVV6chb7fd9H+ypQU45hjgP/7Dv40bziZP9qtzZ5yhUwo89VRq9z939E3rwQdTtxd23Q8apF0qy8v9KtiECXq6dWvqwDK1tanhbMkSDYjnnacTq//gB/7nsbw8NZAHtwtbt+pANStXhrc3yP4fTJuWfp19Ttu9sTNVVcCPf6zzMfYyhjNKjt6aJDXuGM6S59pr/T73REHByllpaeoAIcHRHL/+deCZZ/R040bt2rhjR+q8VXZnNXgMzNe+pjtl9rgfQEPdCy8Ajz/uV8/s9icsnDU2di2cBbs/2rm6ooRVzuz3RS93T+qRXIWzzipngFaEbOVs9GitjN1/v4Zsa8AArVC5XcSCQbm77Puwzz468qD7+Bs2aHC0wcCejhkDXHih7uBHGTxYg16wO+Kpp2q4+Na3NIBYtjJm54O76y6/wjR2rN/OsWP9YFpTo7f5xCd0h98NZ2EBwnbXXLFCg9+IEanVm9//3r8d4Ac/l/0fqKvT/8mWFl0v7uPY6t/8+RooKir8Cb7tiJq//KV/+7vvBm6+2b/sfl7eektDlVs5s8+/fbv+6GGDvJ3XzKqpSV3/s2fr/9/kyf4y+3gDB6YG8qFD/cnLAa3O7r23bp+yCWfz5unrDfs/t68v232jbENcDjCcUXKwchaO4Sx5Hn1Uu40QBdkKlbszUlaWGs6CP2QVFekB7rW1GrLsRKvvv+/fxu7oBcPZ44/rqTt62U9+osHCnXvJbp/dypW7Y5/t8WbB+739tu6sZRI2lL5dB4VcOeurY84ArQitWOEH5aOP1mN2wnZq3Z3iTF3CusJ+PoNdYWtrNRTs2OGHFXcAm5tvBr7xjej30X3vXeXlWnnatk2PX3Ofr6xM18Mpp+iPD7bqNm6cPzLk2LGplbNRozTwrV2bWuULC1aDB+t9FyzQy7ZyZv3DGz/ohBP01K1UAvqe2DAzbJhfORs0KPVxtmzRwUBWrdJuflVV+v/b1qbrbPFify4vQIPxySf7l22gEtH7T5iQHs7sbYYO9Z87GM6ClTM7EIsb8sMqZxs36n0fekiPAwN0PjYrU7dGa+JEP4gGXXihf5ts5OqzngWGM0oOhrNwDGfJYwzfTwoXVhEKVs46OjRAuaOPAf7Opu1W5e5gNTVpaHO7TQH+r+U2hD35JPDcczqfFaDHlgD+9tndWa6q8qtD3a2c7bdf57fPVDnrT+EsqlsjoBUjG8Y7ey/cneKo8NNV9vMaDGduVfSii7R68fWv+8uKirRqFTZCYtgIhi7bPc39saKiQv+efFL/3CreuHH+bSsr/cqZPbVhY4kzpWFdXfrw7kVF+hm2/xP2mLOgY4/V2wRHBqyt9f9vgpUz933dutXv8nnSSfo89ntj6ND06TCCFTwbvCor/ecLq5zZ29rKUli3RvtYdu46wD9GzX08WzlzuzUOGOC31a3QZ1M5y+QrX9H1ke0E8gxnRN3AcBaO4Sx5GM4oSlQ4cy93dOjB/O7IjIAfeubMSV0+YIDucJ13Xuqobu5w1HPnatezL35RB5SYPl134u08TWHhrLjY37HtyTFnnXGrdXEMZ8Hj5LJhX1dxsf+abTgrLk5/3SNH+lWhzt6Lnu4Uh/nRj/Sz4478CKQGiEMP1WqQ2x3OKi4Grr7aH1L93/7N/5Ehivs49v+jokLXkx0Yx+3mN3KkHnM3bZoO828/h/Y0OCKgfbwHHtCq2h136P+dex/AH60xqLJS3/vggDDuyJNu5ay8PPXY0i1btMvm3nvrcYTu4CRh77FIagCxt3G780VVzurq/NudeaYe02a5A4Icfri/3A1FbpfVXXfVrphu18WSkvT1kG2o6ilbZWa3RqJuYDgLx3CWPAxnFMVuB7vSrdGy4cydnBbQnbqtW9OHA1+/3l/21lsa0LZs0VHkBg/W7kR2ItywcAb4VYWujEjY1XAW126NtjtWd46Ls/dxX58NZ4MHp4+q6A5Zno9wNny4jvgYDClul7POJin/7//WChHQ+SAxgAaRUaP0M3HkkbrMhjNAK2vuay0u1kE6/vIX/cEiqnLmqqjQx/jb3/THjZ/8RJe71d+oypldF8H/DbdbZ12dVpM2btTlBx2kP4xMmKCV75df1nnn7PNY7nv8wAM6GIl9jZZdD27bsqmcBdXWAieeqJORu4MEuc/lVs722sufBsAdfMO+7q99TYN4Nt0ac+G113TKgD48PrWk85sQxYS7w8EdVx/DWfIwnFEUG4IyDQgSFc7c0FNUpLcrL9edz6VLU7trAam/XK9d63eDtF0dJ07UeazcdgXD2aWX6mh5XekyZHeSsj1WJKxbY9h6KjQzZuiIet3ZCbWvy+0SaSsAYfPCuSGks+P/gpOa9yb3eCF3WP8oNsB98pPZPf5++2lQsZ/ligo/uNjuhO+840+07ApWzmzQC3aTDOOGUPeYs6FD/R887H2D3VrdkGArV21t+r4UFQFXXaUjKD72mF531FH+81huOPvyl8PbGDbUvHtsmPv8dXXR8w1WV+v/3bXXhs8BF3xcN5C55+++W9+LCy7ofBCgXJoyxZ/3rI8wnFFysHIWjuEseRjOKEpY5SzbcObulJ98sh6v8vOf66h9zz2ny//0Jx2E4+c/T71vY6MfzuyO0+67a2hraooOZ5/5jE6km82xY653382+W1PYUPpxqJzV1kbvOHfGvq6wyllYVemII/zz2XYxDasU5Zo7yl823co+/3kdKTR4nFeU22/X7pw33eQ/x+9/r10i7SiB++4bft9ddtFKjx1CvrRUBwBxf8SIarO9zxVX6GPY2+2+ux/O7P9wsHLm/m+7A224odkGKhEdlRLovFtjUFh3Pncofff5hw2L7n7r/s/bx3SPPXMfz1bO7PO6PxqcfHLqgCUJxnBGyREcHpoUw1nydHRw6ggKF1U5c3foorYFbuXsiit0ctnRo1NHBt1//9SgB+hOVGOjBrHycj8E2KrH+ef7Uz+EDSBx8MGdvqw0++yT/W3jOpR+T2QKZ2EVjqiqSpTGxvTPQW9wu19mO8G1e1xTZ2xV0oaOwYP1+Cx3rrco1dU6J6C77iZPzi6cHX00sGyZX2W2n8dx47QbnStTOHPDqxvO7PPuv7//f+0GuWxGR81UOQuGM3taXa3PvXy5/o8uXJj+uBs3pr8mt3I2erR+VidP7v6k5jHHcEbJwcpZOIaz5GHljKJkc8xZ1A9Z7k7YXnuF75zV1qZXXiZNAt58U3dKhw/3d6hsOJsxQ4dlB3I3ul9XxHW0xp4oLtZKhtslLlM4A3QOqT/8IbsupmEDWPSWSy8N71aYS8HPRbaCx8HtuadOdWK5I2S6RFIn+LaDsYS9N5m6NUZVzlpa9PTUU/1lZWUa1Do6sgvW9v88bJTP4IAg9nTTJv+7KSpYhR0z6lbORHRQkWy7LScQwxklB8NZOIaz5GE4oyg96dbodktydxLdHfGamvRwtttuGs4++CC169Q+++jO65o1OlAIkJ9wFtdujT1VVpb6+ux7GjVYxtln61+hsVXX3mT/P9raevY4wS6Q2Y7wd/rpwBNPANddp8dWuYKfUfdyVOXMDurjhjNAfzwJTlcQdN99OtiPDYzutiQYzg48UEevPPpo/zbdqXYFJxi3E3/3UwxnlBwcECQcw1nyMJxRlGznOesKW0kpL9dftsPmyAI0nNmR4QD9xf+RR3RwBhvOujMsfE/1x8oZkB7ObAU1qnLWn9n/j54eHnHWWTp4zlVX6eVsw9mQIToKJKCVNzckBsOOO71AVZW23Q4IYv32t8CsWenHco4cmTpXWJgzz9TTW27RU/d/1m4L7Oty290TbuWMGM4oQVg5C8dwljwMZxQlqltjcJ6zKC++mN7tyFbOwiovf/yjVsd+8xt97uAoavY++aychQ2lH4fRGnuqrCy1S5yI7kwznKWzg1Bkc6xZJqWlwJVXdj2cudwfOILuuy91YmwRrZ6tWZP6f7nnnv5E8K4bbuh6ddD9/zn2WO3+euCBXXuMzgRHgeznOM8ZJQcrZ+EYzpKH4Yyi9GQofUBHdgsOG21/LbeP4e4EnnVW+oS6rkILZ/25cgbogBNjx+anPYXsS1/SOcHOOis3jzdtmp4GjxfrqTPPTA8w9rizbILNAQcA9fXZPVfYCKtlZdr1NdcDdbByloKVM0oOd2eVO64+hrPk4WiNFKUnx5xFseHM7qQFdwLdcBYc3r4QwpnbLau/jNYI6GsLhrPnnuMOcBgR4JBDcvd4994L3HZb34w2aI87y3XVKWr6i97AcJaC4YySg+EjHMNZ8rByRlHCKmf2WKPiYr2+q+HMdmu0O2nBY86qq/3z+++fel0hhDN2a/T15SiL/VlpafiohL2hK5WzrrDfMX0xZcLo0cCRR6bPf9ZPMZxRcrByFo7hLHkYzihKVOXMVlFaWnpeOQuGM3eH/4ADUq+zx7tt3pz6GH2pv3ZrHDAg2a+PVG9Vzs4/H5g7F7j88tw+bpjycn+ie2I4owThMWfhGM6Sh+GMomTq1lhRoeGsq/MHBStndoffztPkhp+wARAGD85v5cztWmYrZ/b/J8nh5dprs5tsmArfQw9FVz17q3JWWakDkFCfYzij5GDlLBzDWfIwnFGUsGOpSkq06+EeewB33tn1Y2vscSA2WIkAjz2WPqdT1IS7hRDOiop03djn7w/dGk85Jd8toFw5+eTo68aO1c81u6wmBsMZJQfDWTiGs+RhOKMowdBRWqrh5IYbdFLZ7ozUZ4fHP+MMf9nnP596m0WLoncO8x3O7PO64aw/dGuk/uGss3QuQVZJE6PTcCYiAwE8D2CAd/uZxpgfi8gEADMADAXwBoCzjTE7erOxRBkxnIVjOEuejg6+nxQu2K3RngaHuO+KYcOAjRsz/zK/++7R1w0eDCxbpufzMQk1oKFs587+1a2R+oeysvSBeCjWstlKbgfwWWPM/gAOADBVRA4G8DMANxhjJgLYBOC8XmslUTY4tHg4hrPkMYafdwoXFc56qqam+8GqECpntu3B50/yUPpEFEudbmmNavIulnp/BsBnAcz0lt8D4NTeaCBR1lg5C8dwljzs1khRgt0aC6EyNGSIVq2A/HZrBPyQ9tprwGWX5a+SR0QUIatjzkSkGNp1cSKAWwEsBtBojGnzbrICwKheaSFRthjOwjGcJQ/DGUXprcpZT9i5zoD8hzN7+qlP6R8RUYHJ6icjY0y7MeYAAKMBHARgcrZPICIXiMgcEZnT0NDQvVYSZYPhLBzDWfIwnFEUhrNwtkLGShkRFbgubaWMMY0AngFwCIBqEbGVt9EAVkbc53ZjTL0xpr7OzsVA1BsYzsIxnCUPwxlFCQ6lz3CW+rz5en4ioix1Gs5EpE5Eqr3z5QCOBTAfGtK+5N3sHAAP9VIbibLDSajDMZwlD8MZRSnEY84YzoiIspbNMWcjAdzjHXdWBOABY8zDIvI+gBkici2AuQD+0IvtJOocK2fhGM6Sh0PpUxR2awzHbo1EFBOdhjNjzDsADgxZvgR6/BlRYeDOajiGs+ThUPoUJWwS6nwrL/fPs3JGRJQRf0Ki5GDlLBzDWfKwWyNFCVbOCqFb44AB/vl8TkKdz+cnIsoSt1KUHDzmLBzDWfIwnFGUQuzW6IYzVs6IiDJiOKPkYOUsHMNZ8jCcUZRCHK2xEMKZrZgxnBFRgWM4o+RgOAvHcJY8DGcUpdC7Nea7csZujURU4LiVouRgOAvHcJY8HK2RohTigCCFFM5YOSOiAsdwRsnBY87CMZwlD0drpCiFeMyZW73Ld7dGVs6IqMBxK0XJwfARjuEseditkaKwchaOlTMiigmGM0oOdmsMx3CWPAxnFIXHnIVjOCOimGA4o+RgOAvHcJY8DGcUhZWzcOzWSEQxwa0UJQfDWTiGs+RhOKMohT6Ufr4noWbljIgKHMMZJQcHBAnHcJY8HK2RotjKWUmJnrJbY+rzsnJGRAWOWylKDlbOwjGcJQ8rZxTFhrPiYkCkMCpnhTBaIytnRBQTDGeUHAxn4RjOkodD6VMUN5wNHgwMGZLf9gCFUTmzFTOGMyIqcCX5bgBRzjB8hGM4Sx5WziiKDe1FRcATTwCTJuW3PUBhhDP7vCL5eX4ioiwxnFFy8JizcAxnycNwRlHsdrC4GDj44Py2xSqUcFZUxHBGRAWP3RopOditMRzDWfIwnPUJEZkqIh+IyCIRuSzk+rEi8oyIzBWRd0TkeOe6y737fSAiX+izRttwVkghpBDCWVERBwMholhg5YySg+EsHMNZ8jCc9ToRKQZwK4BjAawA8LqIzDLGvO/c7EcAHjDG/FZEpgB4FMB47/w0AHsD2BXAkyKyhzGmvdcbbj8XhRRE3AFB8jmUPo83I6IYKKCtN1EPMZyFYzhLHg6l3xcOArDIGLPEGLMDwAwApwRuYwBUeuerAKzyzp8CYIYxZrsx5iMAi7zH632FWDkrhFDEcEZEMcFwRsnBcBaO4Sx5OFpjXxgFYLlzeYW3zHUlgLNEZAW0avadLty3dxRi5awQ2GPOiIgKHLdUlBzcWQ3HcJY87NZYKM4AcLcxZjSA4wH8UUSy/l4VkQtEZI6IzGloaMhNiwqxclYIiopYOSOiWGA4o+Rg5Swcw1nyMJz1hZUAxjiXR3vLXOcBeAAAjDGvABgIYFiW94Ux5nZjTL0xpr6uri43rWblLBwrZ0QUE9xSUXIwnIVjOEsehrO+8DqASSIyQUTKoAN8zArcZhmAYwBARPaChrMG73bTRGSAiEwAMAnAa33SalbOwvGYMyKKCY7WSMnBcBaO4Sx5GM56nTGmTUS+DeAxAMUA7jTGzBORqwHMMcbMAvA9AL8XkYuhg4Oca4wxAOaJyAMA3gfQBuDCPhmpURuup6wSpWK3RiKKCYYzSg5OQh2O4Sx5GM76hDHmUehAH+6y6c759wEcFnHfnwD4Sa82MAwrZ+HYrZGIYoJbKkoOVs7CMZwlC99PysQYBrMwFRXAkCH5bgURUacYzig5GM7CcWc+Wez7yNFJKUxHB8NZmB/9CHjwwXy3goioU/Hq1jhvHtDYCBwW2ouE+juGj3AMZ8nC95MyMYbd98Lssov+EREVuHhtwX/2M+Dss/PdCipUPOYsHHfmk4XvJ2XCyhkRUazFK5wVFbErD0Vjt8Zw3JlPFr6flAkrZ0REsRavLbgIwxlFYzgLx535ZLHbQL6fFIaVMyKiWIvXMWdFRdwhoWgMZ+EYzpKF7ydlUqiVs/feA9avz3criIgKXvzCGStnFIXhLBx35pOF7ydlUqiVs733zncLiIhioQB/XsuA4Ywy4YAg4bgznyz8EYIyKdTKGRERZSVeW3CGM8qEO6rhGM6SheGMMinUyhkREWWF4YySgzut4RjOkoWfc8qElTMioliL1xacozVSJtxpDcdwliz8nFMmrJwREcVavMIZR2ukTHjMWTi7LvjDRjLwc06ZsHJGRBRr8dqCs1sjZeLulHCn1cfKWbKwckaZsHJGRBRrDGeUHAxn4RjOksV9H7k9pCBWzoiIYi1eW3CGM8rEGP5iHIbhLFlYOaNMWDkjIoq1eIUzDghCmbByFo7hLFkYzigTVs6IiGItXltwDghCmdhfjEX4OXExnCULwxllwsoZEVGsxS+csXJGUdxujdxp9TGcJQtHa6RMWDkjIoq1eG3BGc4oExvOWDlLxXCWLKycUSY89paIKNYYzig57C/GDGepGM6ShaM1UiYdHaycERHFWLy24AxnlAmPOQvHcJYsrJxRJqycERHFWqfhTETGiMgzIvK+iMwTke96y2tF5AkRWeid1vR6a3k8EWXidmskH8NZsjCcUSasnBERxVo2W/A2AN8zxkwBcDCAC0VkCoDLADxljJkE4Cnvcu/iMOmUCY85C8dwliwMZ5QJK2dERLHWaTgzxqw2xrzpnd8KYD6AUQBOAXCPd7N7AJzaS2302XDGro0UhsechWM4SxaO1kiZsHJGRBRrXdqCi8h4AAcCeBXACGPMau+qNQBG5LZpIRjOKBN3fh/utPoYzpKFlTPKhJUzIqJYyzqcicgQAH8F8F/GmC3udcYYAyB0L0FELhCROSIyp6GhoUeNZTijjNitMRzDWbIwnFEmrJwREcVaVltwESmFBrM/GWMe9BavFZGR3vUjAawLu68x5nZjTL0xpr6urq6HrWU4owwYzsIxnCULh9KnTFg5IyKKtWxGaxQAfwAw3xjza+eqWQDO8c6fA+Ch3DcvrTF6yp1MCsNwFo7hLFlYOaNMWDkjIoq1kixucxiAswG8KyJvect+COB6AA+IyHkAlgI4vVda6GLljDJxBwQhH8NZsjCcUSasnBERxVqn4cwY8yKAqC39MbltTicYzigTTkIdjuEsWRjOKBNWzoiIYi1eW3CGs2jt7cDllwPr1+e7Jfnj/mLMnVYfw1mycCh9yoSVMyKiWGM4S4oPPwSuvx548sl8tyR/eMxZOIazZGHljDJh5YyIKNbitQW3vwYynKVrb0897Y84CXU4hrNk4WiNlAkrZ0REsRavcGZ/DeROZjq7k9afd9Z4zFk4hrNkYeWMMmHljIgo1uK1BWe3xmgMZ6ndGsnHcJYsDGeUCStnRESxxnCWFAxnPOYsCsNZsjCcUSasnBERxVq8tuAMZ9EYzhjOojCcJQtHa6RMWDkjIoo1hrOkYDjzBwSx50kxnCULK2eUCStnRESxFq8tOOewisZwxgFBotjPBNdJMjCcUSasnBERxVq8whkrZ9EYztitMQorZ8nCofQpE/sjFRERxRLDWVJwnjOGsygMZ8nCyhll4nbvJiKi2InXFpzhLBorZ6mTUJOP4SxZGM4oE1bOiIhijeEsKRjOeMxZFIazZOFojZQJK2dERLEWry04w1k0hjN2a4zCcJYsrJxRJqycERHFWrzCGUdrjMZwljpKGT8jPoazZGE4o0xYOSMiirV4bcFZOYvGcMbKWRSGs2ThaI19RkSmisgHIrJIRC4Luf4GEXnL+/tQRBqd69qd62b1WaNZOSMiirWSfDegSxjOojGc+ZOvMpylYjhLFlbO+oSIFAO4FcCxAFYAeF1EZhlj3re3McZc7Nz+OwAOdB6ixRhzQB8118fKGRFRrMVrC85wFo3hjJWzKAxnycJw1lcOArDIGLPEGLMDwAwAp2S4/RkA/tInLcuElTMiolhjOEsKO79Zf143bjgjH8NZsjCc9ZVRAJY7l1d4y9KIyDgAEwA87SweKCJzRGS2iJwacb8LvNvMaWhoyE2rWTkjIoq1eG3BOdhDNBvKOAk1K2dBDGfJwqH0C9E0ADONMe4GeJwxph7AVwDcKCK7B+9kjLndGFNvjKmvq6vLTUtYOSMiirV4hTNWzqKxW2PqJNTcafUxnCULK2d9ZSWAMc7l0d6yMNMQ6NJojFnpnS4B8CxSj0frPaycERHFWry24Axn0RjOUn8x5k6rj+EsWRjO+srrACaJyAQRKYMGsLRRF0VkMoAaAK84y2pEZIB3fhiAwwC8H7xvr2DljIgo1jhaY1IwnLFbYxSGs2ThUPp9whjTJiLfBvAYgGIAdxpj5onI1QDmGGNsUJsGYIYxKf9gewH4nYh0QH8Evd4d5bGXG87KGRFRjDGcJQXDGcNZFIazZGHlrM8YYx4F8Ghg2fTA5StD7vcygH17tXFRWDkjIoq1eP28xnAWjeEs9Zgz8jGcJQvDGWXCyhkRUazFawvO44miMZz5vxizcpaK4SxZOFojZcLKGRFRrMUrnLFyFo3znLFbYxSGs2Rh5YwyYeWMiCjW4rUFZziLxsoZw1kUhrNkYTijTFg5IyKKNYazpOAk1H44s+dJMZwlC0drpExYOSMiirV4bcEZzqKxcsZJqKMwnCULK2eUCStnRESxFs9wxh2SdAxnHBAkCsNZsjCcUSasnBERxVq8tuD218D+HECiMJylHnNGPoazZOFojZQJK2dERLEWr3DGbo3RGM44IEgUhrNkYeWMMmHljIgo1uK1BWc4i8ZwxmPOojCcJQvDGWXCyhkRUawxnCUF5zlL3SnhTquP4SxZGM4oE1bOiIhiLV5bcIazaKycsVtjFIazZOFQ+pQJK2dERLEWr3DGqkg0znPGcNYZrpNkYOWMMmHljIgo1uK1BWflLBorZwxnYbgjnzx8TykTVs6IiGKN4SwpGM70tdsBQUhxRz55OJQ+ZcLKGRFRrMVrC85wFo3hjJWzMAxnycP3lDJh5YyIKNYYzpKC4YzhLAx35JOH7yllwsoZEVGsxWsLbr9wuEOSjuHMD2f2PHFHPok4WiNlwsoZEVGsxSuc2S8c7pCk4zxnnIQ6DMNZ8vA9pUxYOSMiirV4bcHZrTEaK2f+L8YMZz7uyCcP31PKhJUzIqJYYzhLCoYzHnMWhjvyycPRGikTVs6IiGItXltwhrNonIQ6NZyRYjhLHr6nlAkrZ0REsdZpOBORO0VknYi85yyrFZEnRGShd1rTu830MJxFY+WMx5yF4Y588vA9pUxYOSMiirVstuB3A5gaWHYZgKeMMZMAPOVd7n0crTEawxmPOQvDHfnk4XtKmbByRkQUa52GM2PM8wA2BhafAuAe7/w9AE7NbbMicLTGaAxnHEo/DHfkk4dD6VMmrJwREcVad7fgI4wxq73zawCMyFF7MmO3xmgMZxwQJAzDWfLwPaVM3B+piIgodnr885oxxgCI3EMQkQtEZI6IzGloaOjZkzGcReM8ZwxnYbgjnzwcrZEy6ehg5YyIKMa6uwVfKyIjAcA7XRd1Q2PM7caYemNMfV1dXTefzsNwFo2Vs9QBQUgxnCUP31PKhJUzIqJY6244mwXgHO/8OQAeyk1zOsEBQaIxnHFAkDDckU8evqeUCStnRESxls1Q+n8B8AqAPUVkhYicB+B6AMeKyEIAn/Mu9z4OCBKN85yxW2MY7sgnD99TyoSVMyKiWCvp7AbGmDMirjomx23pHLs1RmPljOEsDHfkk4ejNVIU+9lg5YyIKLbitQVnOIvGcJY6hDSDiGI4Sx6+pxTFfh5YOSMiii2Gs6RgOOMxZ2G4I588fE8pit3+s3JGRBRb8dqCM5xFYzhjt8Yw3JFPHg6lT1FYOSMiir14hjPukKTjPGep4YwUw1ny8D2lKKycERHFXry24BytMVqwcrZjBzB2LPDgg/lrU19j5Swdd+STh+8pRWHljIgo9uIVztitMVownDU1AcuXAwsX5q9Nfc2dhJo7rYo78snD95SisHJGRBR78dqCs3IWLRjOdu5MPe0P7IAgAHdaLe7IJw+H0qcorJwREcVe/MKZCHdIwgQnod6xI/W0P2C3xnQMZ8nD95SisHJGRBR78duCFxVxhyQMK2cMZ2G4I588HK2RorByRkQUe/EMZ6ycpWM44zFnYdgFLnkYuCkKK2dERLEXvy04uzWGYzhLnYSaFHfkk4fvKUVh5YyIKPbiF85YOQsXnOesP4YzdmtMxx355OF7SlFYOSMiir34bcEZzsJFVc44IEj/xh355GFXVYrCyhkRUewxnCVF2CTUQP+rnNlfjBlEFMNZ8vA9pSisnBERxV78tuAcrTEcjzlLPeaMnxHFHfnk4WiNfUZEporIByKySEQuC7n+BhF5y/v7UEQanevOEZGF3t85fdJgVs6IiGKvJN8N6LJCHRBkzRpg6FCgtDQ/zx+c56w/hjN2a0zHcJY8fE/7hIgUA7gVwLEAVgB4XURmGWPet7cxxlzs3P47AA70ztcC+DGAegAGwBvefTf1aqNZOSMiir34bcELsVvjzp3AyJHAeeflrw2snKWGM1LckU8evqd95SAAi4wxS4wxOwDMAHBKhtufAeAv3vkvAHjCGLPRC2RPAJjaq60FWDkjIkoAhrNcaGnR0z/+MX9tYDhj5SwMd+STh+9pXxkFYLlzeYW3LI2IjAMwAcDTXb1vTrFyRkQUe/HbgncWzn7+c+Dww6Ovf/rp3Ie71tbcPl53cLRGTkIdhjvyycP3tBBNAzDTGNPelTuJyAUiMkdE5jQ0NPS8FaycERHFXjzDWaYdkvnzgXnzwq97913gmGOAxx/PbZsKIZwF5znrj6M1ckCQdNyRTx4Opd9XVgIY41we7S0LMw1+l8as72uMud0YU2+Mqa+rq+thc8HKGRFRAsRvC95Z5ay11e9m2NYGPPKIvzOzYYOerlmT2zYVQjhjt0a/W6M9TwxnScT3tK+8DmCSiEwQkTJoAJsVvJGITAZQA+AVZ/FjAD4vIjUiUgPg896y3sXKGRFR7MUvnHU2WmNLC7B9u97muuuAE08E/vUvvW7bNj3dlOMBsxjOCgOPOUvHHfnk4VD6fcIY0wbg29BQNR/AA8aYeSJytYic7Nx0GoAZxvhvhjFmI4BroAHvdQBXe8t6FytnRESxF7+h9LOpnNnTBQv0/Pr1etrcrKc9CWcvvKDD5R98cPpzAtqdsKys+4/fXQxnPOYsDMNZ8vA97TPGmEcBPBpYNj1w+cqI+94J4M5ea1z4k+opK2dERLGV7HDW1qbn7dxjtnLW2Nj95//BD4DBg4Enn0x/TgBYtw4YPbr7j99dHBAk9ZgzUtyRTx6+pxSFlTMiotiL3xY823DW0uIHFBvOclE5a2rSv7DnBHJ/PFu2OAk1uzWG4Y588vA9pSisnBERxV48K2eZdkjCwpkNLLk45qylJf2Lzw5AAmjlLB9sODNG//rjaI0MZ+m4I588HK2RorByRkQUe/EMZ9lWzmy3RhueclE5swEv7DkBYMuW7j92TwQHCWDlLN+tKQwMZ8nD95SisHJGRBR78QtnnY3W6B5zZoOJDWe5qJx1Fs6CXR77Srsz92lHR/8NZ/YXY+60KndnjeskGThaI0Vh5YyIKPbiF866UjmzXfv6snKWr3DmrhM3nPWXAUHcEMIg4rProbPuwBQfrJxRFFbOiIhiL34/r/WkW2NPR2tsa9PQEwxohRzO+kvljOEsHMNZ8jCcURRWzoiIYi9+W/DOdjJtELOTUQN+eLKVs9bW7k0cbR97587U0OM+1tatXX/cXGA401MOpZ+K4Sx5jAGKi/3zRBYrZ0REsRfPcBZVOevo8Lvxtbb6QSlYOQO617XRvb87QqMNZzU1PaucvfUW8O673btvMJz1t9Ea3RDCypmP4Sx5eGwlRWHljIgo9uK3BRfxK2JB7vKWFn/kxOAxZ0DPw5l7vrUVKCsDqqp6Fs4uvBC4+OLu3TeqcmZM6mAhSWVfP7s1pmI4Sx43nHEofXKxckZEFHvxC2ef/jTwr38Br7+efp3bvbClJbxyZrsD5TqcDRwIDBnSs3C2fj2wYUP37uvupLW3p1bM+kP1jMechWM4S56ODn/nm+8puVg5IyKKvfhtwX/xC2DQIOCuu9KvCx775R5/BmjlbORIPd+dcOZ2ZeyNcNbY2P2RJKMqZ0D/GLEx+Isxd1oVw1nysFsjRWHljIgo9uIXzqqqgAMOAN5+O/06N5ytW+efdytno0bp+VxWzlpa/HDW3QFBjNFw1t2RJKPmOQP6V+WMx5ylYjhLHk62TlFYOSMiir14bsEPPFDDWfB4CzecrV3rn3crZ7vuque7E4I669ZYUdH1ytmWLcBXvwqsWqUVrs2bu3eMWKbKWVLD2ebNwP/8j+6gBo85I8VwljwMZxSFlTMiotiL3yTUgFbOmpuB997TERL/9CcNbMOH+7dZuNA/39uVs550a3zxReDuu4H6en/Zli36urqio8MfydIdrRFIbjh74AHgW98CjjgCGDdOl3GnNRXDWfLYcMb3lIJYOSMiir14hrNPfUpPDznED0nDhgGzZvm3mT9fT8eP1/BkjAa6qioNUbkOZ+Xl3Qtnq1bp6fvv+8s2bepeOCst1REr+0vlbOVK/3TsWD3PcJaK4Sx53MoZR2skFytnRESxF8+f1/bZB/jd74DddvOXDRoUPhn05MlaOWts1B2ZykoNPr01IEhDA7BgQebH2bABeOUVPW8DhhvOutPlsqMDKCnxz/eHAUFssF21iqM1RmE4Sx47IAg/5xTEyhkRUezFdwt+wQU6YXNLC3DRRRpy7Lxm1ogRGsRaWoDXXtNln/xk98NZNsecAcBee0XPxQYA110HHHWUtiusctbdcFZa6p/vD5WzsHDGUexSMZwljx1Kn+GMglg5IyKKvfiGM2vgQD1eq70dmDdPl9kvpvHjtbthSwvw8su6g3rQQbkJZ+6E1jaclTi9RGfPjn6cN97QwDR/vl85c0eX7E7b3MpZf5nnzIazlSs5CXUUhrPk4YAgFIWVMyKi2EvGFnzyZD21w+vX1fmnNpy99BKw335a3aqpAd55J3XQkGwEK2dPPQUce6wOTDJqlI4eaH3/+8CKFemPYQzw1lt6/t13/YDhOvdcYMmSrrUtrFvjgAF6OenhLFO3xrfeAg4+GFi9Oi9NzDuGs+RhOKMorJwREcVePAcECZo8WUPYAw/o5V/9CvjGN4AvfAH4+GNg40bg2WeB731Pr29t1SC1xx7Ad78LnHOOdkNsbgZmzACOPBKYMAH48EPgmGOAXXbRatTrr2vXQRENezfeqFWuc84BfvpTDUGjR+vokbNna7sqK3WwiilTgP3318eyIe699/zKmaupCbjmmvCJtqO0t6d2a9yxQ4/D2769d8NZU5Mea9fXdu70q41h4cy6+Wbg1Vf1GMUrr+zzZuadG87cYyZ7+phx3/lrbweKi/Pdiu5hOKMorJwREcVeMsJZRYWGqlNO0csnnKBByxhg+nRd1tGhw64DwGc+A/zrXzoE+0036Z/rjjtSL5eU+EPU77KLjhL5t79ptezll4E99/Rv++1vAyefrMeQzZihy5YvBx55JDVsDRoE3HuvBgxb3XPdfbdW5oYMASZNAnbf3R9xcts2oK1NA9iwYbqjuWOHXzm75BJ93EGDNDyedRZw0kl6ubkZqK4GFi/W08GD9X4tLRp4Pv1pDbNr1+rzbd2qzzdkiB5LV1qqA5qsX6/dM198UV/zhAl6v9ZWrVAC2s20shKordXXOHCgVrCWL9eqZksLsGyZVjI7OrStgwcDY8bojvOKFXpZRB+3tlbPt7Xpfe3ACK+95k9FIKJtXL5cw/HSpbr8uuuA227TNlRX62MtW6btss/X1KTrp6lJ/zo69LjF8nI9ntFOcTBkiD63MdqWnTv1r6YGKCvT88boei0p0faUlABr1ujr2LhRn6+uTu+zdKleV1Gh79HAgbp8yxZg6FB9L4qKtE2Vlfr8Gzboc48bp4/f0eF3Z3X/bBfZQYOAjz7SdTJ2LPDBB/ojQWWltq29XU+rq3V5U5M+tz12sqpKB7vZtk3bX1Ghy6qq9L7Llul9m5v18W14W7VKX4+IPtfWrfo869frgD623aWlWsmurNTnnzBBH9tOzr5kib5P1dV6202btD0lJXqbQYN0vXR06HopKdF1XFKi627nTn0tra362jZu1B9MVq/W++y9t7Zt2DBdVl6ur3X4cL1fU5NWoktLtS0dHf7/7bhxepstW7QNRUX6/1haqn/GpL8v9q+qSp+7tNSfHmTwYP+zbrnnlyzxL//qV8DDD+v7Yf+fy8v1+Vtb9cef6urUH2p27tR1U1bmf44B/7wxemzujTeCYubZZ/W0vDyvzSAiou4T04e/vNbX15s5c+b03hOsXas7iXaofUDD03//t1bAfvhDXWaMv/M0c6Z/2+3bdZLqtWt1B2/33TVgLF2qOzN77qnLDj5Yd3RHj/arVdlYs0a7Xm7apDucd9+tO6Nf/CLwne/ozuKee2pF7803daeqtVW7YK5erTuxdgexslLPr1+vO1nNzVrBu+UWfxLrM87Q4LJliz5Oa6sfxMaP153Rlha/iiDiH0tXWqrL7E5vY2PqgCtlZboO6+u1rXaQgpISv1JXXBw+oXZpqX+bujrdGS8r00DS2OiPtDl4sLa5o0Nfqzsap338Cy/UEDtihD7uDTfo8rvu0p39FSt0wJiXX9a2bN+uO/Vr1uh6bmnR6mpRkb5O96+9XUNuc7NeHjZM29fU5Fcu3PBlA1NZmR8ibXhra9P3044WKqIBcutW3bkfOVIft6VF/zZs0MdpaPBHJR08WC83NmpQKSrSz2ZHh77m4mI/ENi/8nJdxxdfDNxzj+78L1qkQbuuTt9T+/7bSdDtdBPr1/vBqrFRw9rgwboetm7V29r777abf/sVK/zQtOuuus7b2vQ+VVW6Pqqrdb3b9dfUpP9bzc36+B9+qOtBRNsyYYL+z7W06Odg+HBdj7aKtG2btrG0VNdbe7v+tbX54W/AAG2f/UwvWKA/thQV6fGflZX6vzl0qP+eb9zo3761VZ9n0iR9vG3b9HTpUl0vVVW6TkS0DfYHFPuDQdjfqlX6uux7sP/++rhuF2kgvUJ2yCHa7mef1fY2Nel97I83ZWX63g8fruvFfi7t89ofdOzn2IY9e/4TnwB++cvobVkWROQNY0x957ckIAffj/Pn60jG06YBf/wjq2dERAUs03dkj8KZiEwFcBOAYgB3GGOuz3T7Xg9ncWZDTGddrTrrtmJD0rp1upPrPp4NpE1N/siSrm3bNATW1PgBwjLGP8Zr6FDd0Vu6VEOeDRTV1boTuHixPu/o0XrZDR11dRqkmpp0p7e4OLWbnDF6u+3b9fEsEV1eVKT3sZXMgQMzry8iyguGs67Jyffj888Dhx6aOjgVEREVnEzfkd3egotIMYBbARwLYAWA10VkljHm/cz3pFDZHv/S2a+h9voRI9KvKyvT07BgBmhY2n338OtEtBuna8IE/36DBun54mKtALqqqtIfz21DsPuW+3gudtUhIop25JH5bgEREfVQT/o9HARgkTFmiTFmB4AZAE7JTbOIiIiIiIj6l56Es1EAljuXV3jLUojIBSIyR0TmNDQ09ODpiIiIiIiIkqvXjxg2xtxujKk3xtTX2fnHiIiIiIiIKEVPwtlKAGOcy6O9ZURERERERNRFPQlnrwOYJCITRKQMwDQAs3LTLCIiIiIiov6l26M1GmPaROTbAB6DDqV/pzFmXs5aRkRERERE1I/0aDIUY8yjAB7NUVuIiIiIiIj6rV4fEISIiIiIiIg6x3BGRERERERUABjOiIiIiIiICoAYY/ruyUQaACzt4cMMA7A+B83p77gec4PrMXe4LnOjUNbjOGMMJ7fMUo6+H4HCef/jjusxN7gec4PrMTcKaT1Gfkf2aTjLBRGZY4ypz3c74o7rMTe4HnOH6zI3uB77N77/ucH1mBtcj7nB9ZgbcVmP7NZIRERERERUABjOiIiIiIiICkAcw9nt+W5AQnA95gbXY+5wXeYG12P/xvc/N7gec4PrMTe4HnMjFusxdsecERERERERJVEcK2dERERERESJE6twJiJTReQDEVkkIpfluz2FTETuFJF1IvKes6xWRJ4QkYXeaY23XETkZm+9viMin8hfywuLiIwRkWdE5H0RmSci3/WWc112gYgMFJHXRORtbz1e5S2fICKveuvrfhEp85YP8C4v8q4fn9cXUGBEpFhE5orIw95lrsd+jt+PXcPvyJ7j92Nu8Psxt5Lw/RibcCYixQBuBXAcgCkAzhCRKfltVUG7G8DUwLLLADxljJkE4CnvMqDrdJL3dwGA3/ZRG+OgDcD3jDFTABwM4ELvc8d12TXbAXzWGLM/gAMATBWRgwH8DMANxpiJADYBOM+7/XkANnnLb/BuR77vApjvXOZ67Mf4/dgtd4PfkT3F78fc4PdjbsX++zE24QzAQQAWGWOWGGN2AJgB4JQ8t6lgGWOeB7AxsPgUAPd45+8BcKqz/F6jZgOoFpGRfdLQAmeMWW2MedM7vxX6Dz8KXJdd4q2PJu9iqfdnAHwWwExveXA92vU7E8AxIiJ909rCJiKjAZwA4A7vsoDrsb/j92MX8Tuy5/j9mBv8fsydpHw/ximcjQKw3Lm8wltG2RthjFntnV8DYIR3nus2C17J+0AAr4Lrssu8rgZvAVgH4AkAiwE0GmPavJu46+r/1qN3/WYAQ/u0wYXrRgA/ANDhXR4Krsf+jtud3OB2vZv4/dgz/H7MmRuRgO/HOIUzyiGjw3RyqM4sicgQAH8F8F/GmC3udVyX2THGtBtjDgAwGvpL/+T8tih+ROREAOuMMW/kuy1EScbtevb4/dhz/H7suSR9P8YpnK0EMMa5PNpbRtlba7sQeKfrvOVctxmISCn0i+dPxpgHvcVcl91kjGkE8AyAQ6DdWkq8q9x19X/r0bu+CsCGvm1pQToMwMki8jG069pnAdwErsf+jtud3OB2vYv4/Zhb/H7skcR8P8YpnL0OYJI36koZgGkAZuW5TXEzC8A53vlzADzkLP8PbySlgwFsdrok9Gte/+M/AJhvjPm1cxXXZReISJ2IVHvnywEcCz0+4RkAX/JuFlyPdv1+CcDThpMywhhzuTFmtDFmPHQb+LQx5kxwPfZ3/H7MDW7Xu4Dfj7nB78fcSNT3ozEmNn8AjgfwIbQv7hX5bk8h/wH4C4DVAHZC+9ieB+1L+xSAhQCeBFDr3VagI30tBvAugPp8t79Q/gAcDu2S8Q6At7y/47kuu7we9wMw11uP7wGY7i3fDcBrABYB+F8AA7zlA73Li7zrd8v3ayi0PwCfAfAw1yP/vPea349dW1/8juz5OuT3Y27WI78fc79OY/39KF4DiYiIiIiIKI/i1K2RiIiIiIgosRjOiIiIiIiICgDDGRERERERUQFgOCMiIiIiIioADGdEREREREQFgOGMiIiIiIioADCcERERERERFQCGMyIiIiIiogLw/wFJA4B4xnGtjQAAAABJRU5ErkJggg=="
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1080x360 with 2 Axes>"
            ],
            "image/png": "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"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1080x360 with 2 Axes>"
            ],
            "image/png": "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77xz9PzMmhWfqSOOiLBw002xjn32ief+7W/SVlvFfnvkEenQQ6VPfzoCwbHHRgA64gjpve+Ny7/+Vfr1r6O39Zpr4keApCHxoQ/FsNyNN45Gyrhx8bn++MelN74x9sGtt0rf+176a7gU7+O006Jh9JGPxOfwy1+OYHPTTbHMWWely48bF73HX/1qhIz/+A/pN7+J5z/1VDx+0UWx7PjxsQ2SfXjOOekQos02i1lYv//9uH3RRdImm0R5H3wwgp4U1//+9whCkyb137cf/rD09a9HgPzJTyIYzpkjvfKVERCvvz6We+UrIwCdc470sY9FT/b8+fE+dtghGou77x49dZtvHv+rZ5wRz73yyvSccp/5jPTDH0rHHRc93cuWSZ/7XPz4ct55Eb6OPz6enw11UnzettpK+uxnpRkzokH69NPRW3/MMfHc17xGurA239Kll8aw6mXLInCuWBEh6c47+4el0aOlT30qflh49lnpTW+K75WkN3/ixPg8HHZYfB5OPz3eqxTLXXutdMEF8X12zz1x+apXRdmuuy4mYNpyy/g/OOKIeN5DD8V34h57xA9U668vlMxQhz7W96itWhWfm1YFtfoetUmTYp3N9qjNnh2XSe99X1/87ySTgg2Fe/yNHRvfq9mgJsX10dVtGkniGLVGCGrVU4GgJndvy99rXvMaX6d6e90/9CH3jTaKr+UbbnC/+OLkK9r94x8f2npWrHDfZx/3n/40bs+c6X7bbf2XmT3b/be/dZ8zJ72vpyd9Lcl9woT4e+Ur3T/96bhv663Tx//5n937+twPPdT985+PdVxxRfr4+PHur351XP/hD927u92//W33X/3K/aST3F98MV7TPd735Mnuo0a5f/az7mecEc8bMyYuZ850nz+//3tYtCi2WU+P+wsvuB9yiPsjj/Rf5o473LfYwn3KFPe5c/s/NmtWrPu1r83fjg8/7P773695/6pV8b4XLnQfNy7W8fvfu7///VHu7u789bm7r17tft99cX2vvaJskvumm7pvs43761/vfvvtcd8ZZ7h/7GNxfdw49y23dH/oIfeDD063cVeX+8tfHvtp223d11sv7t9yS/e3vCVdbsyY9HM1YUK63Lbbum+2Wbrcu96VLjeUv513jstzznH/p39K78+u0yz26WGHDb6+jTeO7fCOd8TtN77R/Wc/i+3t7n788emy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"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {}
    }
  ]
}