import os
import numpy as np
import pandas as pd
from io import StringIO
import astropy.wcs as pywcs
import src.utils as utils
import src.config as C
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



class SKADataset:
    """
    SKA dataset wrapper.
    """

    def __init__(self, train_set_path=None, test_set_path=None, subset=2.0):
        
        # Save training and test2set path+patch_dim-1s
        self.train_set_path = train_set_path
        self.test_set_path = test_set_path

        # Save column names
        self.col_names = ['ID',
              'RA (core)',
              'DEC (core)',
              'RA (centroid)',
              'DEC (centroid)',
              'FLUX',
              'Core frac',
              'BMAJ',
              'BMIN',
              'PA',
              'SIZE',
              'CLASS',
              'SELECTION',
              'x',
              'y']
    
        # Process the training set
        self.raw_train_df = None
        self.proc_train_df = pd.DataFrame()
        if self.train_set_path is not None:
            assert os.path.exists(
                self.train_set_path
            ), "Missing SKA training set .txt file"
            self.train_df_path = f"{os.path.splitext(self.train_set_path)[0]}.pkl"
            self.raw_train_df = self._load_dataset(
                self.train_set_path, self.train_df_path
            )
            if subset < 1.0:
                self.raw_train_df = self._get_portion(self.raw_train_df, subset)

    def _load_dataset(self, dataset_path, dataframe_path):
        """
        Loads the SKA dataset into a Pandas DataFrame,
        starting from a specifically-formatted txt file.
        """
        # Load the DataFrame, if it was already pickled before
        if os.path.exists(dataframe_path):
            try:
                return pd.read_pickle(dataframe_path)
            except ValueError:
                pass

        # Otherwise load the txt file and extract data
        else:
            df = self._prepare_dataset(dataset_path)

        # Save the dataframe into a pickle file
        df.to_pickle(dataframe_path)
        
        return df


    def _get_portion(self, df, subset=1.0):
        """
        Returns a random subset of the whole dataframe.
        """
        amount = int(df.shape[0] * subset)
        random_indexes = np.random.choice(
            np.arange(df.shape[0]), size=amount, replace=False
        )
        return df.iloc[random_indexes].reset_index(drop=True)

    def _prepare_dataset(self, dataset_path):

        df = pd.read_csv(dataset_path, skiprows=18, header=None, names=self.col_names, delimiter=' ', skipinitialspace=True)

        return df

    
    def _convert_boxes_to_px_coord(self, boxes_dataframe, fits_header):
        # convert all boxes into pix coords
        # boxes_dataframe = ska_dataset.raw_train_df
        fits_header = fits_header #data_560Mhz_1000h_fits[0].header
        image_width = fits_header['NAXIS1']
        image_height = fits_header['NAXIS2']
        pixel_res_x_arcsec = abs(float(fits_header['CDELT1'])) * 3600
        pixel_res_y_arcsec = abs(float(fits_header['CDELT2'])) * 3600

        coords={
        'x1':[],
        'y1':[],
        'x2':[],
        'y2':[],
        'major_semia_px':[],
        'minor_semia_px':[],
        'pa_in_rad':[],
        'width':[],
        'height':[]
        }

        # remove useless data from dataframe
        # filt_dataframe = boxes_dataframe[['x', 'y', 'RA (centroid)', 'DEC (centroid)', 'BMAJ', 'BMIN', 'PA']]
        wc = pywcs.WCS(fits_header)

        for idx, box in boxes_dataframe.iterrows():
        # compute centroid coord to check with gt data
            cx, cy = wc.wcs_world2pix([[box['RA (centroid)'], box['DEC (centroid)'], 0, 0]], 0)[0][0:2]

            # Safety check
            dx = cx - box['x']
            dy = cy - box['y']
            if (dx >= 0.01 or dy >= 0.01):
                raise ValueError("Computed Centroid is not valid")

            if (cx < 0 or cx > image_width):
                print('got it cx {0}, {1}'.format(cx, fits_fn))
                raise ValueError("Out of image BB")
            if (cy < 0 or cy > image_height):
                print('got it cy {0}'.format(cy))
                raise ValueError("Out of image BB")

            major_semia_px = box['BMAJ'] / pixel_res_x_arcsec / 2 #actually semi-major 
            minor_semia_px = box['BMIN'] / pixel_res_x_arcsec / 2 #actually semi-major 
            pa_in_rad = np.radians(box['PA']) # ATTENTION: qui dovrebbe essere 180°-box[PA]

            x1, y1, x2, y2 = utils._get_bbox_from_ellipse(pa_in_rad, major_semia_px, minor_semia_px, cx, cy, image_height, image_width)

            coords['x1'].append(x1)
            coords['y1'].append(y1)
            coords['x2'].append(x2)
            coords['y2'].append(y2)
            coords['major_semia_px'].append(major_semia_px)
            coords['minor_semia_px'].append(minor_semia_px)
            coords['pa_in_rad'].append(pa_in_rad)
            coords['width'].append(abs(x2-x1))
            coords['height'].append(abs(y2-y1))

        return coords

    def _extend_dataframe(self, df, cols_dict):
      df_from_dict = pd.DataFrame.from_dict(cols_dict)
      if df.shape[0] != df_from_dict.shape[0]:
          raise ValueError("Dimension of DataFrame and dict passed don't match!")
      return pd.concat([df, df_from_dict], axis=1)



    def _split_in_patch(self, img, df, img_name, x_origin, y_origin, patch_dim=200, is_multiple=False):
        h, w = img.shape
        fits_filename = img_name.split('/')[-1].split('.')[0]
        
        patches = {
            "patch_name": [],
            "patch_xo": [],
            "patch_yo": [],
            "patch_dim": [],
            "gt_id": []
        }

        patches_json = {}

        # Add new columns to df
        df['x1s'] = None
        df['y1s'] = None
        df['x2s'] = None
        df['y2s'] = None
        
        if (w % patch_dim !=0 or h % patch_dim != 0) and is_multiple:
            raise ValueError('Image size is not multiple of patch_dim. Please choose an appropriate value for patch_dim.')

        for i in range(0, h, patch_dim):
            if i <= 1000:
                for j in range(0, w, patch_dim):                    
                    patch_xo = x_origin+j
                    patch_yo = y_origin+i
                    gt_id = []

                    gt_id = self._find_gt_in_patch(patch_xo, patch_yo, patch_dim, df)

                    if len(gt_id) > 0:
                        perc = 95
                        filename = f'{fits_filename}_{patch_xo}_{patch_yo}'
                        img_patch = img[i:i+patch_dim, j:j+patch_dim]
                        percentileThresh = np.percentile(img_patch, perc)

                        # Cut bboxes that fall outside the patch
                        df_scaled = df.loc[df['ID'].isin(gt_id)].apply(self._cut_bbox, patch_xo=patch_xo, patch_yo=patch_yo, patch_dim=patch_dim, axis=1)
                        
                        df_scaled["patch_name"] = filename
                        df_scaled["patch_xo"] = patch_xo
                        df_scaled["patch_yo"] = patch_yo
                        df_scaled["patch_dim"] = patch_dim

                        self.proc_train_df = self.proc_train_df.append(df_scaled)
                        patch_id = str(patch_xo)+str(patch_xo)
                        self._save_bbox_files(img_patch, patch_id, df_scaled)
                            
                        print(df_scaled.columns)
                        print(df_scaled.head())
                        return #TODO: remove this


                        # Create figure and axes
                        fig, ax = plt.subplots()

                        # Display the image
                        plt.imshow(img_patch * (1.0 / percentileThresh))
                        for box_index in gt_id:
                            box = df_scaled.loc[df_scaled['ID']==box_index].squeeze()
                            plt.gca().add_patch(Rectangle((box.x1-patch_xo, box.y1-patch_yo), box.x2 - box.x1, box.y2-box.y1,linewidth=.1,edgecolor='r',facecolor='none'))
                            plt.text(box.x-patch_xo, box.y-patch_yo, box_index, fontsize=1)
                        
                        plt.show()
        
        return #patches
    
    def _cut_bbox(self, x, patch_xo, patch_yo, patch_dim):
        
        x.x1 = max(x.x1, patch_xo)
        x.y1 = max(x.y1, patch_yo)
        x.x2 = min(x.x2, patch_xo+patch_dim)
        x.y2 = min(x.y2, patch_yo+patch_dim)
        x.x1s = x.x1 - patch_xo
        x.y1s = x.y1 - patch_yo
        x.x2s = x.x2 - patch_xo
        x.y2s = x.y2 - patch_yo

        
        return x

    def _save_bbox_files(self, img_patch, patch_id, df):
        if not os.path.exists(os.path.join(C.TRAIN_PATCHES_FOLDER, f"{patch_id}")):
            os.makedirs(os.path.join(C.TRAIN_PATCHES_FOLDER, f"{patch_id}/"))

        np.save(os.path.join(C.TRAIN_PATCHES_FOLDER, f"{patch_id}/{patch_id}.npy"), img_patch)
        df.to_pickle(os.path.join(C.TRAIN_PATCHES_FOLDER, f"{patch_id}/{patch_id}.pkl"))
        print('image saved')
        
        return
        

    
    def _find_gt_in_patch(self, patch_xo, patch_yo, patch_dim, gt_df):

        def filter_func(x, patch_xo=patch_xo, patch_yo=patch_yo, patch_dim=patch_dim):
            return x.x >= patch_xo and x.y >= patch_yo and x.x <= patch_xo + patch_dim and x.y <= patch_yo + patch_dim

        filtered_df = gt_df[gt_df.apply(filter_func, axis = 1)]

        idx2 = filtered_df['ID'].tolist()
        return idx2