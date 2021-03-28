import os

# Generic config
RANDOM_SEED = 42
DATA_SUBSET = 0.1
DOWNLOAD_FOLDER = 'data/training'
DATA_FOLDER = os.path.join(os.getcwd(), "data")
TRAIN_DATA_FOLDER = os.path.join(DATA_FOLDER, "training")
TRAIN_SET_PATH = os.path.join(TRAIN_DATA_FOLDER, "training_SKA.txt")

required_files = [
    {
        "file_name": "training_SKA.txt",
        "url": "https://owncloud.ia2.inaf.it/index.php/s/iTOVkIL6EfXkcdR/download" #54Mb
    },
    # {
    #     "file_name": "PrimaryBeam_B1.fits",
    #     "url": "https://owncloud.ia2.inaf.it/index.php/s/ZbaSDe7zGBYgxL1/download" #300Kb
    # },
    # {
    #     "file_name": "SynthesBeam_B1.fits",
    #     "url": "https://owncloud.ia2.inaf.it/index.php/s/cwzf1BO2pyg9TVv/download" #4Gb
    # },
    {
        "file_name": "560Mhz_1000h.fits",
        "url": "https://owncloud.ia2.inaf.it/index.php/s/hbasFhd4YILNkCr/download" #4Gb
    }
]