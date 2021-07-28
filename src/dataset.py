import os
from IPython.core.display import display
import itertools
import numpy as np
import scipy.stats as st
import pandas as pd
import astropy.wcs as pywcs
from astropy.io import fits
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import src.utils as utils
import src.config as config


class SKADataset:
    """
    SKA dataset wrapper.
    Schema:

    1. load dataset
    1.5. load FITS image
    2. preprocess
    3. split train/val
    """

    def __init__(self):

        # Save training and test set
        self.train_set_path = None
        self.test_set_path = None

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
        self.cleaned_train_df = pd.DataFrame()
        self.proc_train_df = pd.DataFrame()

        for download_info in config.required_files:
            if not os.path.exists(os.path.join(config.TRAIN_DATA_FOLDER, download_info['file_name'])):
                utils.download_data(download_info['file_name'], download_info['url'], config.DOWNLOAD_FOLDER)



    def load_dataset(self, train_set_path=config.TRAIN_SET_PATH, subset=config.DATA_SUBSET):


        """
        Loading portion <subset> of given dateset
        """

        def _load_dataset(dataset_path, dataframe_path):
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
                df = _prepare_dataset(dataset_path)

            # Save the dataframe into a pickle file
            df.to_pickle(dataframe_path)

            return df

        def _get_portion(df, subset=1.0):
            """
            Returns a random subset of the whole dataframe.
            """
            amount = int(df.shape[0] * subset)
            random_indexes = np.random.choice(
                np.arange(df.shape[0]), size=amount, replace=False
            )
            return df.iloc[random_indexes].reset_index(drop=True)

        def _prepare_dataset(dataset_path):

            df = pd.read_csv(dataset_path, skiprows=18, header=None, names=self.col_names, delimiter=' ', skipinitialspace=True)

            return df


        # Save training and test set
        self.train_set_path = train_set_path

        if self.train_set_path is not None:
            assert os.path.exists(
                self.train_set_path
            ), "Missing SKA training set .txt file"
            self.train_df_path = f"{os.path.splitext(self.train_set_path)[0]}.pkl"
            self.raw_train_df = _load_dataset(
                self.train_set_path, self.train_df_path
            )
            if subset < 1.0:
                self.raw_train_df = _get_portion(self.raw_train_df, subset)

        print(f'Dataset shape: {self.raw_train_df.shape}')
        display(self.raw_train_df.head())

    def load_train_image(self, image_path=config.IMAGE_PATH, primary_beam=None, print_info=False, load_pb=False, preprocess_image=True):
        fits_image = fits.open(image_path)
        if print_info:
            print(fits_image.info())

        print(f'Loading FITS file {fits_image.filename()}')

        self.image_filename = fits_image.filename()
        self.image_data = fits_image[0].data[0,0]
        self.image_header = fits_image[0].header

        if load_pb:
            fits_image = fits.open(image_path)
            self.primary_beam_data =  primary_beam[0].data[0,0]
            self.primary_beam_header =  primary_beam[0].header

        #TODO: add plot function

    def process_dataset(self, use_pb=False, b='b5', fancy=True):
        """
        TODO: write desc
        """
        def _apply_primary_beam_correction(b, box, area_pixel):
            # Taken form https://github1s.com/ICRAR/skasdc1/blob/HEAD/scripts/create_train_data.py
            num_sigma = 0.05

            if b == 'b5':
                b5_sigma = 3.8465818166348711e-08
                b5_median = -5.2342429e-11
                b_n_sigma = b5_median + num_sigma * b5_sigma

            if b == 'b1':
                # these are for training map only
                b1_sigma = 3.8185009938219004e-07 #
                b1_median = -1.9233363e-07 #
                b_n_sigma = b1_median + num_sigma * b1_sigma

            pbwc = pywcs.WCS(self.primary_beam_header)

            total_flux = float(box['FLUX'])
            total_flux = _primary_beam_gridding(pbwc, total_flux, box['RA (centroid)'], box['DEC (centroid)'], self.primary_beam_data)
            total_flux /= area_pixel
            return total_flux, b_n_sigma


        def _primary_beam_gridding(pbwc, total_flux, ra, dec, primary_beam):
            # Taken form https://github1s.com/ICRAR/skasdc1/blob/HEAD/scripts/create_train_data.py
            # to understand see http://www.alma.inaf.it/images/Imaging_feb16.pdf
            x, y = pbwc.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
            pb_data = primary_beam[0].data[0][0]
            pbv = pb_data[int(y)][int(x)]
            return total_flux / pbv

        def _enlarge_bbox(x, scale_factor):

            if (1 < x.width < 2) | (1 < x.height < 2):
                x.width = x.width * scale_factor
                x.height = x.height * scale_factor
                x.x1 = x.x - x.width/2
                x.y1 = x.y - x.height/2
                x.x2 = x.x + x.width/2
                x.y2 = x.y + x.height/2
            elif (x.width <= 1) | (x.height <= 1):
                x.width = x.width * 6
                x.height = x.height * 6
                x.x1 = x.x - x.width/2
                x.y1 = x.y - x.height/2
                x.x2 = x.x + x.width/2
                x.y2 = x.y + x.height/2

            return x

        def _extend_dataframe(df, cols_dict):
            df_from_dict = pd.DataFrame.from_dict(cols_dict)
            if df.shape[0] != df_from_dict.shape[0]:
                raise ValueError("Dimension of DataFrame and dict passed don't match!")
            return pd.concat([df, df_from_dict], axis=1)


        wc = pywcs.WCS(self.image_header)

        image_width = self.image_header['NAXIS1']
        image_height = self.image_header['NAXIS2']
        pixel_res_x_arcsec = abs(float(self.image_header['CDELT1'])) * 3600
        # pixel_res_x_arcsec = 1.02168000000E-05*3600
        # print(pixel_res_x_arcsec)
        pixel_res_y_arcsec = abs(float(self.image_header['CDELT2'])) * 3600
        BMAJ = float(self.image_header['BMAJ']) #2.53611124208E-05
        BMIN = float(self.image_header['BMIN']) #2.53611124208E-05
        psf_bmaj_ratio = (BMAJ / pixel_res_x_arcsec) * 3600.0
        psf_bmin_ratio = (BMIN / pixel_res_y_arcsec) * 3600.0
        g = 2 * pixel_res_x_arcsec # gridding kernel size as per specs
        g2 = g ** 2

        coords={
        'x1':[],
        'y1':[],
        'x2':[],
        'y2':[],
        'major_semia_px':[],
        'minor_semia_px':[],
        'pa_in_rad':[],
        'width':[],
        'height':[],
        'area_orig':[],
        'area_cropped':[]
        }

        #rows with selection == 0 must be deleted (from 274883 rows to 190553)
        boxes_dataframe = self.raw_train_df[self.raw_train_df.SELECTION != 0]

        #list  of rows to be deleted due to a too weak flux
        id_to_delete = []
        faint = 0
        faint_a = 0

        for idx, box in tqdm(boxes_dataframe.iterrows(),total=boxes_dataframe.shape[0]):
        # compute centroid coord to check with gt data
            cx, cy = wc.wcs_world2pix([[box['RA (centroid)'], box['DEC (centroid)'], 0, 0]], 0)[0][0:2]

            # Safety check
            dx = cx - box['x']
            dy = cy - box['y']

            if (dx >= 0.01 or dy >= 0.01):
                raise ValueError("Computed Centroid is not valid")
            if (cx < 0 or cx > image_width):
                print('got it cx {0}, {1}'.format(cx))#, fits_fn))
                raise ValueError("Out of image BB")
            if (cy < 0 or cy > image_height):
                print('got it cy {0}'.format(cy))
                raise ValueError("Out of image BB")

            bmaj = box['BMAJ']
            bmin = box['BMIN']

            if (fancy):
                #TODO use w1, w2 to replace bmaj, bmin
                if (2 == box['SIZE']):
                    if (box['CLASS'] in [1, 3]): # extended source
                        # don't change anything
                        w1 = bmaj
                        w2 = bmin
                    else:
                        # assuming core is fully covered by source
                        w1 = bmaj / 2.0 #+ g / 10.0
                        w2 = w1 #keep it circular Gaussian
                else: # compact source [1 or 3]
                    # 1,2,3 for SS-AGNs, FS-AGNs, SFGs
                    # 1,2,3 for LAS, Gaussian, Exponential
                    if (1 == box['CLASS'] and 1 == box['SIZE']):
                        w1 = bmaj / 2.0
                        w2 = bmin / 2.0
                    elif (3 == box['CLASS'] and 3 == box['SIZE']):
                        w1 = bmaj
                        w2 = bmin
                    else:
                        raise Exception('unknown combination')

                #TODO calculate b1 and b2 from w1 and w2 for gridded sky model
                b1 = np.sqrt(w1 ** 2 + g2) * psf_bmaj_ratio
                b2 = np.sqrt(w2 ** 2 + g2) * psf_bmin_ratio
            else:
                b1 = bmaj
                b2 = bmin

            # major_semia_px = box['BMAJ'] / pixel_res_x_arcsec / 2
            # minor_semia_px = box['BMIN'] / pixel_res_x_arcsec / 2

            major_semia_px = b1 / pixel_res_x_arcsec / 2
            minor_semia_px = b2 / pixel_res_x_arcsec / 2
            pa_in_rad = np.radians(box['PA']) # TODO: ATTENTION: qui dovrebbe essere 180°-box[PA]

            x1, y1, x2, y2 = utils._get_bbox_from_ellipse(pa_in_rad, major_semia_px, minor_semia_px, cx, cy, image_height, image_width)

            area_pixel = abs(x2-x1) * abs(y2-y1)
            if(config.clean_dataset) & (area_pixel<=0.):
                id_to_delete.append(idx)
                faint_a += 1
                continue

            if use_pb:
                total_flux, b_n_sigma = _apply_primary_beam_correction(idx, b, area_pixel)
                if (total_flux < b_n_sigma):
                    id_to_delete.append(idx)
                    faint += 1
                    continue

            orig_area = abs(x2-x1) * abs(y2-y1)

            # crop it around the border
            xp_min = max(x1, 0)
            yp_min = max(y1, 0)
            xp_max = min(x2, image_width - 1)
            if (xp_max <= xp_min):
                break
            yp_max = min(y2, image_height - 1)
            if (yp_max <= yp_min):
                break
            new_area = (yp_max - yp_min) * (xp_max - xp_min)

            if (orig_area / new_area > 4):
                print('cropped box is too small, discarding...')
                break

            coords['x1'].append(x1)
            coords['y1'].append(y1)
            coords['x2'].append(x2)
            coords['y2'].append(y2)
            coords['major_semia_px'].append(major_semia_px)
            coords['minor_semia_px'].append(minor_semia_px)
            coords['pa_in_rad'].append(pa_in_rad)
            coords['width'].append(abs(x2-x1))
            coords['height'].append(abs(y2-y1))
            coords['area_orig'].append(orig_area)
            coords['area_cropped'].append(new_area)



        print(f'Initial dataset shape: {boxes_dataframe.shape}')
        print(f'Found {faint_a} boxes with zero area')
        print(f'Rows to be deleted: {len(id_to_delete)}')
        cleaned_df = copy.copy(boxes_dataframe.drop(index = id_to_delete).reset_index(drop=True))
        print(f'New dataset shape: {cleaned_df.shape}')
        print('Extending dataset with new computed columns...')
        cleaned_df = copy.copy(_extend_dataframe(cleaned_df, coords))
        print(f'Final cleaned dataset shape: {cleaned_df.shape}')
        print()

        if config.enlarge_bbox:
            print('Enlarging bboxes...')
            self.cleaned_train_df= copy.copy(cleaned_df.apply(_enlarge_bbox, scale_factor = config.bbox_scale_factor, axis=1))
            print('DONE - Enlarging bboxes...')
        return

    def cut_preprocess_image(self, plot_Noise_Hist = False):
        self.x_origin = int(np.floor(min(self.cleaned_train_df['x1'])))
        self.y_origin = int(np.floor(min(self.cleaned_train_df['y1'])))

        x2_max = int(np.floor(max(self.cleaned_train_df['x2'])))
        y2_max = int(np.floor(max(self.cleaned_train_df['y2'])))

        self.data_560Mhz_1000h_train = self.image_data[self.y_origin:y2_max, self.x_origin :x2_max]
        data_flat = self.data_560Mhz_1000h_train.flatten()

        neg_values = data_flat[data_flat < 0]
        max_val = max(data_flat)
        print('database len', len(self.cleaned_train_df))
        print('min x val =',self.x_origin )
        print('min y val =',self.y_origin )
        print('x len =',(x2_max - self.x_origin) )
        print('y len =',(y2_max - self.y_origin) )
        print('max val =',max_val )

        # # histogram of noise (- noise, + noise)
        if plot_Noise_Hist:
        # # we know that negative values are due to noise, and we assume a gaussian noise distribution
            min_val = min(data_flat)
            print('min val =',min_val )
            # plt.hist(data_flat, bins = 40, range = (0.0001, 0.006))
            plt.hist(data_flat, bins = 40, range = (abs(min_val), max_val))

        clipped_img = np.clip(self.data_560Mhz_1000h_train, a_min=0, a_max=np.max(self.image_data))

        mu, std = self._compute_halfgaussian_noise(np.abs(neg_values), True)
        self._Delete_sources_low_signal(self.cleaned_train_df, clipped_img, std)

        self._compute_RGB_scale(clipped_img, std, max_val)

        return

    def _Delete_sources_low_signal(self, gt_df, img, std):

        print(img.shape)
        high_snr_ID =[]
        self.cleaned_train_df['HighSTD'] = None

        def filter_func(x1, y1, x2, y2):
            # print('x1', x1,'y1', y1,'x2', x2,'y2', y2)
            img_box = img[ x1 -self.x_origin: x2 - self.x_origin ,y1 - self.y_origin : y2 -self.x_origin ].copy()
            data_flat = img_box.flatten()
            if(len(data_flat)< 1):
                return False
            # if(np.mean(data_flat)> 2.5*std):
            #     plt.imshow(img_box, cmap='viridis')

            return np.mean(data_flat)> 2.5*std

        #self.cleaned_train_df['HighSTD'] = df_scaled.loc[self.cleaned_train_df['ID'].isin(gt_id)].apply(self._from_image_to_patch_coord, patch_xo=patch_xo, patch_yo=patch_yo, axis=1)
        print(len(self.cleaned_train_df))
        for ind, row in self.cleaned_train_df.iterrows():
            high_snr_ID.append(filter_func(int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])))

           #high_snr_ID.add(filter_func(row['x1'], row['y1'], row['x2'], row['y2']))
        #self.cleaned_train_df['HighSTD'] = self.cleaned_train_df.apply(lambda row : filter_func(row['x1'], row['y1'],
        # row['x2'], row['y2']), axis=1)
        #self.cleaned_train_df.drop(high_snr_ID)
        self.cleaned_train_df= self.cleaned_train_df.iloc[high_snr_ID, :]
        #self.cleaned_snr_df = self.cleaned_train_df[high_snr_ID]
        #self.cleaned_train_df = self.cleaned_snr_df
        #print('list length', len(high_snr_ID))
        print(sum(high_snr_ID))
        print("new df",len(self.cleaned_train_df))
        #print(img.size())
        return

    def _compute_RGB_scale(self, clipped_img, std, max_val):

        thresh_low = std * 2.5
        min_magnitude_order = int(np.log10(thresh_low)) -1
        max_magnitude_order = int(np.log10(max_val))

        # print('lower threshold',thresh_low)
        # print('min magnitude',  min_magnitude_order)
        # print('max magnitude', max_magnitude_order)

        one_magnitude_range = int(np.floor(256/(max_magnitude_order - min_magnitude_order)))
        lndelta = np.linspace(0., 1., one_magnitude_range)
        self.training_image = self._img_from_float_to_RGB(clipped_img, lndelta, one_magnitude_range, min_magnitude_order, max_magnitude_order)
        self.pixel_mean = np.repeat(int(np.mean(self.training_image)), 3)
        return

    def _img_from_float_to_RGB(self, img, lndelta, one_magnitude_range, min_magnitude_order, max_magnitude_order):
        rgb_val = 0
        thrsld_min = np.power(10., min_magnitude_order)
        thrsld_max = np.power(10., max_magnitude_order)

        img[ img<thrsld_min ] = rgb_val
        img[ img>thrsld_max ] = 255

        delta_ord_mag = max_magnitude_order - min_magnitude_order

        for rng in range(delta_ord_mag):
            real_deltas = np.power( 10., lndelta ) * np.power(10., min_magnitude_order + rng)
            for inner_rng in range(one_magnitude_range-1):
                img[ ((img > real_deltas[inner_rng]) & (img < real_deltas[inner_rng+1])) ] = rgb_val
                rgb_val += 1

        return img

    def _compute_halfgaussian_noise(self, data, plot=False):
        # fit dist to data
        mu, std = st.halfnorm.fit(data)

        print(mu, std)

        if plot:
            plt.hist(data, bins = 40, range = (0., max(data)))
            # Plot the PDF.
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = st.halfnorm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
        return mu, std

    def generate_patches(self, limit, plot_patches = False):
        self.patch_list = {}
        self.patch_list = self._split_in_patch( config.patch_dim, show_plot=plot_patches, limit=limit)
        return

    def _split_in_patch(self, patch_dim=100, is_multiple=False, show_plot=False, limit=None):

        self.cleaned_train_df
        h, w = self.training_image.shape
        fits_filename = self.image_filename.split('/')[-1].split('.')[0]

        # Add new columns to df
        self.cleaned_train_df['x1s'] = None
        self.cleaned_train_df['y1s'] = None
        self.cleaned_train_df['x2s'] = None
        self.cleaned_train_df['y2s'] = None
        self.cleaned_train_df['class_label'] = None

        if (w % patch_dim !=0 or h % patch_dim != 0) and is_multiple:
            raise ValueError('Image size is not multiple of patch_dim. Please choose an appropriate value for patch_dim.')

        patches_list = []
        for i in tqdm(range(0, h, int(patch_dim/2))):
            if i<= limit*patch_dim:                           #TODO: remove this
                for j in range(0, w, int(patch_dim/2)):
                    if j <=  limit*patch_dim :   #TODO: remove this
                        patch_xo = self.x_origin+j
                        patch_yo = self.y_origin+i
                        gt_id = []

                        gt_id = self._find_gt_in_patch(patch_xo, patch_yo, patch_dim, self.cleaned_train_df)

                        if len(gt_id) > 0:
                            filename = f'{fits_filename}_{patch_xo}_{patch_yo}'
                            img_patch = self.training_image[i:i+patch_dim, j:j+patch_dim].copy()

                            # Cut bboxes that fall outside the patch
                            df_scaled = self.cleaned_train_df.loc[self.cleaned_train_df['ID'].isin(gt_id)].apply(self._cut_bbox, patch_xo=patch_xo, patch_yo=patch_yo, patch_dim=patch_dim, axis=1)
                            df_scaled = df_scaled.loc[self.cleaned_train_df['ID'].isin(gt_id)].apply(self._from_image_to_patch_coord, patch_xo=patch_xo, patch_yo=patch_yo, axis=1)

                            df_scaled["patch_name"] = filename
                            df_scaled["patch_xo"] = patch_xo
                            df_scaled["patch_yo"] = patch_yo
                            df_scaled["patch_dim"] = patch_dim
                            df_scaled['bbox_w'] = np.abs(df_scaled['x2'] - df_scaled['x1'])
                            df_scaled['bbox_h'] = np.abs(df_scaled['y2'] - df_scaled['y1'])

                            df_scaled = df_scaled.reset_index(drop=True)
                            df_scaled['ID'] = df_scaled['ID'].astype(int).astype('object')
                            df_scaled['SIZE'] = df_scaled['SIZE'].astype(int).astype('object')
                            df_scaled['CLASS'] = df_scaled['CLASS'].astype(int).astype('object')
                            df_scaled['SELECTION'] = df_scaled['SELECTION'].astype(int).astype('object')
                            df_scaled['class_label'] = df_scaled[['SIZE', 'SELECTION']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1)

                            patch_index = i * (h // patch_dim) +j

                            self.proc_train_df = self.proc_train_df.append(df_scaled)
                            patch_id = str(patch_index)+'_'+str(patch_xo)+'_'+str(patch_yo)+'_'+str(patch_dim)
                            self._save_bbox_files(img_patch, patch_id, df_scaled)
                            patches_list.append(patch_id)

                            if show_plot:
                                plt.imshow(img_patch, cmap='viridis', vmax=255, vmin=0)
                                print('max gray level val = ', img_patch.max())
                                for box_index in gt_id:
                                    box = df_scaled.loc[df_scaled['ID']==box_index].squeeze()
                                    plt.gca().add_patch(Rectangle((box.x1-patch_xo, box.y1-patch_yo), box.x2 - box.x1, box.y2-box.y1,linewidth=.1,edgecolor='r',facecolor='none'))
                                    plt.text(box.x-patch_xo, box.y-patch_yo, box_index, fontsize=1)
                                plt.show()

        # # TODO: trasfromare in attributo della classe dataset
        self.class_list = self.proc_train_df['class_label'].unique()
        print(self.class_list)
        self.num_classes = len(self.proc_train_df['class_label'].unique())
        print(self.num_classes)
        return patches_list

    def _cut_bbox(self, x, patch_xo, patch_yo, patch_dim):

        x.x1 = max(x.x1, patch_xo)
        x.y1 = max(x.y1, patch_yo)
        x.x2 = min(x.x2, patch_xo+patch_dim-1)
        x.y2 = min(x.y2, patch_yo+patch_dim-1)

        x = self._from_image_to_patch_coord(x, patch_xo, patch_yo)

        return x

    def _from_image_to_patch_coord(self, x, patch_xo, patch_yo):

        x.x1s = x.x1 - patch_xo
        x.y1s = x.y1 - patch_yo
        x.x2s = x.x2 - patch_xo
        x.y2s = x.y2 - patch_yo

        return x

    def _save_bbox_files(self, img_patch, patch_id, df_scaled):
        if not os.path.exists(os.path.join(config.TRAIN_PATCHES_FOLDER, f"{patch_id}")):
            os.makedirs(os.path.join(config.TRAIN_PATCHES_FOLDER, f"{patch_id}/"))

        np.save(os.path.join(config.TRAIN_PATCHES_FOLDER, f"{patch_id}/{patch_id}.npy"), img_patch)
        df_scaled.to_pickle(os.path.join(config.TRAIN_PATCHES_FOLDER, f"{patch_id}/{patch_id}.pkl"))
        print(f'Patch {patch_id} saved')
        return

    def _find_gt_in_patch(self, patch_xo, patch_yo, patch_dim, gt_df):

        def filter_func(x, patch_xo=patch_xo, patch_yo=patch_yo, patch_dim=patch_dim):
            return x.x >= patch_xo and x.y >= patch_yo and x.x <= patch_xo + patch_dim and x.y <= patch_yo + patch_dim

        filtered_df = gt_df[gt_df.apply(filter_func, axis = 1)]

        idx2 = filtered_df['ID'].tolist()
        return idx2

    def analyze_class_distribution(self):

        # number of possible class combinations in each patch, -1: because if there isn't any class the patch is not saved
        self.num_combinations = pow(2, self.num_classes) - 1

        patch_class_list = []
        patch_class_bool = []
        patch_class_int = []
        self.patch_list_per_class = {}

        for j in range(1,self.num_combinations+1):
            key_j = 'key_{}'.format(j)
            self.patch_list_per_class[key_j] = []

        for patch_id in self.patch_list:
            img_data_path = os.path.join(config.TRAIN_PATCHES_FOLDER, patch_id, f"{patch_id}.pkl")
            img_data_patch = pd.read_pickle(img_data_path)

            patch_class_list = img_data_patch['class_label'].unique()

            for class_idx in self.class_list:
                if class_idx in patch_class_list:
                    patch_class_bool.append('1')
                else:
                    patch_class_bool.append('0')

            patch_class_2=''.join(patch_class_bool)
            b = (int(patch_class_2, base=2))
            patch_class_int.append(b)
            self.patch_list_per_class['key_{}'.format(b)].append(patch_id)
            #print (b, patch_class_list)
            patch_class_bool.clear()

        res =[]
        for idx in range(1, self.num_classes+1):
            #res.append((bin(idx)))
            res.append(int(bin(idx)[2:]))

        print('number of possible class combinations:',self.num_combinations )
        #print('current combinations:',patch_class_int.unique() )
        # print(self.patch_list_per_class)
        self.class_distribution = patch_class_int
        plt.hist(patch_class_int )
        #plt.legend(['columns >= 4 has 'f"{self.class_list[0]}",  'columns ... has'f"{self.class_list[1]}", 'odd columns has'f"{self.class_list[2]}" ])
        plt.xlabel("classes and combinations")
        plt.ylabel("num patches")
        plt.title("Histogram")
        plt.show()
        return

    def split_train_val_stratification(self, random_state, val_portion=.2):

        self.train_patch_list = []
        self.val_patch_list = []

        if val_portion==0:
            self.val_patch_list = []
            self.train_patch_list = list(itertools.chain.from_iterable([[] + i for i in self.patch_list_per_class.values()]))

        for idx in range(1, self.num_combinations+1):
            try:
                train, val = train_test_split(self.patch_list_per_class['key_{}'.format(idx)],
                 test_size = val_portion, random_state=random_state)

                self.train_patch_list = self.train_patch_list + train
                self.val_patch_list = self.val_patch_list + val
            except:
                print('key_{}'.format(idx),"not splitted")
                continue
        #TODO: aggiungere trasloco val patch nella loro folder
        print('split ended')

        return

    def split_train_val(self, random_state, val_portion=.2):
        # self.train_patch_list_no_strat = []
        # self.val_patch_list_no_strat = []
        train_class_distribution =[]

        train, val = train_test_split(self.patch_list,  test_size = val_portion, random_state=random_state)

        for idx in range(1, self.num_combinations+1):
            common_ID = set(train) & set(self.patch_list_per_class['key_{}'.format(idx)])

            if(len(self.patch_list_per_class['key_{}'.format(idx)])!=0):
                print("len of class:", format(idx), len(self.patch_list_per_class['key_{}'.format(idx)]))
                print("len of common ID:", len(common_ID))
                train_class_distribution.append((len(common_ID)/len(self.patch_list_per_class['key_{}'.format(idx)]))*100)
            else:
                train_class_distribution.append(0)

        x_val= list(range(1, len(self.patch_list_per_class.keys())+1))
        # print("x", x_val)
        # print("y", train_class_distribution)
        # print("name",self.patch_list_per_class.keys() )
        plt.bar(x_val, train_class_distribution, tick_label =list(self.patch_list_per_class.keys()), width = 0.8)
        #plt.legend(['columns >= 4 has 'f"{self.class_list[0]}",  'columns ... has'f"{self.class_list[1]}", 'odd columns has'f"{self.class_list[2]}" ])
        plt.xlabel("classes ratio")
        plt.ylabel("num patches")
        plt.title("Histogram")
        plt.show()