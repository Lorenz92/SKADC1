import os
import numpy as np
import pandas as pd
from io import StringIO
import astropy.wcs as pywcs
import src.utils as utils
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# these values are copyed form ICRAR code https://github1s.com/ICRAR/skasdc1/blob/HEAD/scripts/create_train_data.py
b5_sigma = 3.8465818166348711e-08
b5_median = -5.2342429e-11
num_sigma = 0.05
b5_three_sigma = b5_median + num_sigma * b5_sigma

class SKADataset:
    """
    SKA dataset wrapper.
    """

    def __init__(self, train_set_path=None, test_set_path=None, subset=1.0):
    
        # Save training and test set paths
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

    # copied form https://github1s.com/ICRAR/skasdc1/blob/HEAD/scripts/create_train_data.py
    def _primary_beam_gridding(self, total_flux, ra, dec, pb_wcs, pb_data):
        x, y = pb_wcs.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
        #print(pb_data.shape)
        pbv = pb_data[int(y)][int(x)]
        #print(x, y, pbv)
        return total_flux * pbv

    def _convert_boxes_to_px_coord(self, boxes_dataframe, fits_header):
        # convert all boxes into pix coords
        # boxes_dataframe = ska_dataset.raw_train_df
        image_width = fits_header['NAXIS1']
        image_height = fits_header['NAXIS2']
        pixel_res_x_arcsec = abs(float(fits_header['CDELT1'])) * 3600
        pixel_res_y_arcsec = abs(float(fits_header['CDELT2'])) * 3600

        coords={
        #'ID':[],    
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

        #clean_boxes_dataframe = pd.DataFrame().reindex_like(boxes_dataframe)

        # remove useless data from dataframe
        # filt_dataframe = boxes_dataframe[['x', 'y', 'RA (centroid)', 'DEC (centroid)', 'BMAJ', 'BMIN', 'PA']]
        wc = pywcs.WCS(fits_header)

        #list  of rows to be deleted due to a too weak flux
        id_to_delete = []
        faint = 0
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

            #area_pixel = major_semia_px * minor_semia_px *4
            # if(area_pixel==0):
            #     #id_to_delete.append(box['ID'])
            #     faint += 1
            #     continue
            # copied form https://github1s.com/ICRAR/skasdc1/blob/HEAD/scripts/create_train_data.py
            # total_flux = float(box['FLUX'])
            # total_flux = self._primary_beam_gridding(total_flux, box['RA (centroid)'],  box['DEC (centroid)'], pb_wcs, primaryBeam_data)
            # total_flux /= area_pixel
            # if (total_flux < b5_three_sigma):
            #      #id_to_delete.append(box['ID'])
            #     faint += 1
            #     continue

            x1, y1, x2, y2 = utils._get_bbox_from_ellipse(pa_in_rad, major_semia_px, minor_semia_px, cx, cy, image_height, image_width)

            #coords['ID'].append(box['ID'])
            coords['x1'].append(x1)
            coords['y1'].append(y1)
            coords['x2'].append(x2)
            coords['y2'].append(y2)
            coords['major_semia_px'].append(major_semia_px)
            coords['minor_semia_px'].append(minor_semia_px)
            coords['pa_in_rad'].append(pa_in_rad)
            coords['width'].append(abs(x2-x1))
            coords['height'].append(abs(y2-y1))

        # boxes_dataframe.iloc[:rows_to_delete]
        # boxes_dataframe['id_check'] = coords['ID']
        # boxes_dataframe['x1']= coords['x1']
        # boxes_dataframe['y1']= coords['y1']
        # boxes_dataframe['x2']= coords['x2']
        # boxes_dataframe['y2']= coords['y2']
        
        return coords
        

    def remove_rows_by_flux(self, df, fits_header, primaryBeam):
        pbhead = primaryBeam[0].header
        pb_wcs = pywcs.WCS(pbhead)
        primaryBeam_data = primaryBeam[0].data[0][0]

        def filter(df_row, fits_header=fits_header, pb_wcs=pb_wcs, primaryBeam_data=primaryBeam_data):

            image_width = fits_header['NAXIS1']
            image_height = fits_header['NAXIS2']
            pixel_res_x_arcsec = abs(float(fits_header['CDELT1'])) * 3600
            pixel_res_y_arcsec = abs(float(fits_header['CDELT2'])) * 3600

            major_semia_px = df_row['BMAJ'] / pixel_res_x_arcsec / 2 #actually semi-major 
            minor_semia_px = df_row['BMIN'] / pixel_res_x_arcsec / 2 #actually semi-major 
            pa_in_rad = np.radians(df_row['PA']) # ATTENTION: qui dovrebbe essere 180°-box[PA]

            area_pixel = major_semia_px * minor_semia_px *4
            if(area_pixel==0):
                return False

            # copied form https://github1s.com/ICRAR/skasdc1/blob/HEAD/scripts/create_train_data.py
            total_flux = float(df_row['FLUX'])
            total_flux = self._primary_beam_gridding(total_flux, df_row['RA (centroid)'],  df_row['DEC (centroid)'], pb_wcs, primaryBeam_data)
            total_flux /= area_pixel
            if (total_flux < b5_three_sigma):
                return False

            return True

        df[df.apply(filter, axis=1)]
        #print(df.apply(filter, axis=1))
        #print(df[df.apply(filter, axis = 1)])


    def _extend_dataframe(self, df, cols_dict):
      # aggiungere x1... al df
      df_from_dict = pd.DataFrame.from_dict(cols_dict)
      if df.shape[0] != df_from_dict.shape[0]:
          raise ValueError("Dimension of DataFrame and dict passed don't match!")
      return pd.concat([df, df_from_dict], axis=1)



    def split_in_patch(self, img, df, img_name, patch_dim=200):
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
        
        if w % patch_dim !=0 or h % patch_dim != 0:
            raise ValueError('Image size is not multiple of patch_dim. Please choose an appropriate value for patch_dim.')

        #TODO: fix this taking into account that origin should be in upper-left corner
        for i in range(0, h, patch_dim):
            print('----')
            for j in range(0, w, patch_dim):
                print(f'riga:{i}', f'colonna:{j}')
                patch_xo = 16000+j
                patch_yo = 16000+i

                patch = {}
                gt_id = []
                img_patch = img[i:i+patch_dim, j:j+patch_dim]
                patch = {
                    "orig_coords": img_patch.tolist(),
                    # "scaled_coords": scaled_img_patch
                }

                gt_id = self._find_gt_in_patch(patch_xo, patch_yo, patch_dim, df)
                
                if len(gt_id) > 0:
                    perc = 95
                    percentileThresh = np.percentile(img_patch, perc)       

                    # Create figure and axes
                    fig, ax = plt.subplots()

                    # Display the image
                    plt.imshow(img_patch * (1.0 / percentileThresh))
                    for box_index in gt_id:
                        print(box_index)
                        box = df.iloc[box_index]
                        plt.gca().add_patch(Rectangle((box.x1-patch_xo, box.y1-patch_yo), box.x2 - box.x1, box.y2-box.y1,linewidth=1,edgecolor='r',facecolor='none'))
                    
                    plt.show()
                    input("Press Enter to continue...")
                    return

                    
                
                filename = f'{fits_filename}_{patch_xo}_{patch_yo}'
                patches_json[filename] = patch

                patches["patch_name"].append(filename)
                patches["patch_xo"].append(patch_xo)
                patches["patch_yo"].append(patch_yo)
                patches["patch_dim"].append(patch_dim)
                patches["gt_id"].append(gt_id)


        with open(f'data/training/{fits_filename}.json', 'w', encoding='utf-8') as f:
            json.dump(patches_json, f, ensure_ascii=False, indent=4)

        
        return patches
        
    
    def _find_gt_in_patch(self, patch_xo, patch_yo, patch_dim, gt_df):
        def filter_func(x, patch_xo=patch_xo, patch_yo=patch_yo, patch_dim=patch_dim):
            # print(f'xo:{patch_xo}')
            # print(f'yo:{patch_yo}')
            # print(f'x1:{x.x1}')
            # print(f'y1:{x.y1}')
            # print(f'x2:{x.x2}')
            # print(f'y2:{x.y2}')
            return x.x1 >= patch_xo and x.y1 >= patch_yo and x.x2 <= patch_xo + patch_dim and x.y2 <= patch_yo + patch_dim

        # filter = lambda x: x.x1 >= patch_xo and x.y1 >= patch_yo and x.x2 <= patch_xo + patch_dim and x.y2 <= patch_yo + patch_dim
        
        filtered_df = gt_df[gt_df.apply(filter_func, axis = 1)]
        if filtered_df.shape[0] > 1:
            print(f'patch_xo:{patch_xo}, patch_yo:{patch_yo}')
            print(filtered_df.head())

        idx = filtered_df.index.tolist()
        return idx