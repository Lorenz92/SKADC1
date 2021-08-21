import os
from os import listdir
from IPython.core.display import display
import itertools
import numpy as np
import scipy.stats as st
from scipy.special import erfinv
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
    TODO: scrivere meglio qui
    """

    def __init__(self, k, print_info=False, use_pb=False):
        
        # Save training and test set
        self.train_set_path = config.TRAIN_SET_PATH_CLEANED
        # self.train_set_path = config.TRAIN_SET_PATH
        self.train_df_path = f"{os.path.splitext(self.train_set_path)[0]}.pkl"

        self.test_set_path = None
        self.subset = config.DATA_SUBSET
        self.image_path = config.IMAGE_PATH
        self.primary_beam_path = config.PRIMARY_BEAM_PATH

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

        self.coords={
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
    
        # Process the training set
        self.cleaned_train_df = pd.DataFrame()
        self.proc_train_df = pd.DataFrame()

        # Constant used for dataset cleaning
        mad2sigma = np.sqrt(2) * erfinv(2 * 0.75 - 1)

        # Internal preprocessing methods
        
        def load_dataset():
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
            
            if self.train_set_path is not None:
                assert os.path.exists(
                    self.train_set_path
                ), "Missing SKA training set .txt file"
                raw_train_df = _load_dataset(
                    self.train_set_path, self.train_df_path
                )
                if self.subset < 1.0:
                    raw_train_df = _get_portion(raw_train_df, self.subset)

            print(f'Dataset shape: {raw_train_df.shape}')
            display(raw_train_df.head())
            
            return raw_train_df

        def load_train_image(print_info=False, use_pb=False):
            fits_image = fits.open(self.image_path)
            if print_info:
                print(fits_image.info())

            print(f'Loading FITS file {fits_image.filename()}')

            image_filename = fits_image.filename()
            image_data = fits_image[0].data[0,0]
            image_header = fits_image[0].header
            
            if use_pb:
                primary_beam = fits.open(self.primary_beam_path)  
                self.primary_beam_data =  primary_beam[0].data[0,0]
                self.primary_beam_header =  primary_beam[0].header

            #TODO: add plot function

            return image_filename, image_data, image_header

        def process_dataset(use_pb=False, k=2.5, fancy=True):
            """
            TODO: write desc
            """
            def _apply_primary_beam_correction(k, box, area_pixel, x1, y1, x2, y2):
                # Taken form https://github1s.com/ICRAR/skasdc1/blob/HEAD/scripts/create_train_data.py
                
                # bbox_region = self.image_data[int(np.floor(y1)):int(np.floor(y2)), int(np.floor(x1)):int(np.floor(x2))]

                # med = np.nanmedian(bbox_region)
                # mad = np.nanmedian(np.abs(bbox_region - med))
                # sigma = mad / mad2sigma

                # threshold = med + k * sigma
                # print('threshold:', threshold)

                threshold = k*255.0E-9

                pbwc = pywcs.WCS(self.primary_beam_header)

                total_flux = float(box['FLUX'])
                total_flux = _primary_beam_gridding(pbwc, total_flux, box['RA (centroid)'], box['DEC (centroid)'], self.primary_beam_data)
                # total_flux /= area_pixel
                return total_flux, threshold


            def _primary_beam_gridding(pbwc, total_flux, ra, dec, primary_beam_data):
                # Taken form https://github1s.com/ICRAR/skasdc1/blob/HEAD/scripts/create_train_data.py
                # to understand see http://www.alma.inaf.it/images/Imaging_feb16.pdf
                x, y = pbwc.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
                pb_data = primary_beam_data
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

            
            #rows with selection == 0 must be deleted (from 274883 rows to 190553)
            boxes_dataframe = self.raw_train_df[self.raw_train_df.SELECTION != 0] 

            #list  of rows to be deleted due to a too weak flux
            id_to_delete = []
            faint = 0
            faint_a = 0

            for idx, box in tqdm(boxes_dataframe.iterrows(),total=boxes_dataframe.shape[0]):
            # compute centroid coord to check with gt data
                cx, cy = self.wc.wcs_world2pix([[box['RA (centroid)'], box['DEC (centroid)'], 0, 0]], 0)[0][0:2]

                # Safety check
                dx = cx - box['x']
                dy = cy - box['y']

                if (dx >= 0.01 or dy >= 0.01):
                    raise ValueError("Computed Centroid is not valid")
                if (cx < 0 or cx > self.image_width):
                    print('Got cx {0}, {1}'.format(cx))
                    raise ValueError("Out of image BB")
                if (cy < 0 or cy > self.image_height):
                    print('Got cy {0}'.format(cy))
                    raise ValueError("Out of image BB")
                
                bmaj = box['BMAJ']
                bmin = box['BMIN']

                if (fancy):
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
                    b1 = np.sqrt(w1 ** 2 + self.g2) * self.psf_bmaj_ratio
                    b2 = np.sqrt(w2 ** 2 + self.g2) * self.psf_bmin_ratio
                else:
                    b1 = bmaj
                    b2 = bmin

                major_semia_px = b1 / self.pixel_res_x_arcsec / 2
                minor_semia_px = b2 / self.pixel_res_x_arcsec / 2
                pa_in_rad = np.radians(box['PA']) # TODO: ATTENTION: qui dovrebbe essere 180°-box[PA]
                    
                x1, y1, x2, y2 = utils._get_bbox_from_ellipse(pa_in_rad, major_semia_px, minor_semia_px, cx, cy, self.image_height, self.image_width)

                area_pixel = abs(x2-x1) * abs(y2-y1)
                
                if(config.clean_dataset) & (area_pixel<=0.):
                    id_to_delete.append(idx)
                    faint_a += 1
                    continue
                
                #TODO: remove this and related variables after the last cleaning attempt
                if use_pb:
                    # print('Applying primary beam correction...\n')
                    total_flux, threshold = _apply_primary_beam_correction( k, box, area_pixel, x1, y1, x2, y2)
                    if (total_flux < threshold):
                        id_to_delete.append(idx)
                        faint += 1
                        continue
                
                orig_area = abs(x2-x1) * abs(y2-y1)
                
                # crop it around the border
                xp_min = max(x1, 0)
                yp_min = max(y1, 0)
                xp_max = min(x2, self.image_width - 1)
                if (xp_max <= xp_min):
                    break
                yp_max = min(y2, self.image_height - 1)
                if (yp_max <= yp_min):
                    break
                new_area = (yp_max - yp_min) * (xp_max - xp_min)

                if (orig_area / new_area > 4):
                    print('cropped box is too small, discarding...')
                    break

                self.coords['x1'].append(x1)
                self.coords['y1'].append(y1)
                self.coords['x2'].append(x2)
                self.coords['y2'].append(y2)
                self.coords['major_semia_px'].append(major_semia_px)
                self.coords['minor_semia_px'].append(minor_semia_px)
                self.coords['pa_in_rad'].append(pa_in_rad)
                self.coords['width'].append(abs(x2-x1))
                self.coords['height'].append(abs(y2-y1))
                self.coords['area_orig'].append(orig_area)
                self.coords['area_cropped'].append(new_area)
                


            print(f'Initial dataset shape: {boxes_dataframe.shape}')
            print(f'Found {faint_a} boxes with zero area')
            print(f'Rows to be deleted: {len(id_to_delete)}')       
            cleaned_df = copy.copy(boxes_dataframe.drop(index = id_to_delete).reset_index(drop=True))
            print(f'New dataset shape: {cleaned_df.shape}')
            print('Extending dataset with new computed columns...')
            cleaned_df = copy.copy(_extend_dataframe(cleaned_df, self.coords))
            print(f'Final cleaned dataset shape: {cleaned_df.shape}')
            print()

            if config.enlarge_bbox:
                print('Enlarging bboxes...')
                cleaned_train_df= copy.copy(cleaned_df.apply(_enlarge_bbox, scale_factor = config.bbox_scale_factor, axis=1))
                print('DONE - Enlarging bboxes...')
            
            display(cleaned_train_df.head())
            return cleaned_train_df
        
        def preprocess_train_image(plot_noise=False):
            """
            TODO: write desc
            """
            def _convert_to_RGB(image, data, max_val):
                print('Removing positive noise and rescaling to 0-255 interval...')
                mu, stdev = _compute_halfgaussian_noise(data, True)
                thresh_low = stdev * 2.5
                min_magnitude_order = int(np.log10(thresh_low)) -1
                max_magnitude_order = int(np.log10(max_val))

                one_magnitude_range = int(np.floor(256/(max_magnitude_order - min_magnitude_order)))
                lndelta = np.linspace(0., 1., one_magnitude_range)
                training_image = _RGB_to_FLOAT(image, lndelta, one_magnitude_range, min_magnitude_order, max_magnitude_order)
                pixel_mean = np.repeat(int(np.mean(training_image)), 3)
                return training_image, pixel_mean

            def _RGB_to_FLOAT(img, lndelta, one_magnitude_range, min_magnitude_order, max_magnitude_order):
                rgb_val = 0
                thrsld_min = np.power(10., min_magnitude_order)
                thrsld_max = np.power(10., max_magnitude_order)

                img[img < thrsld_min] = rgb_val
                img[img > thrsld_max] = 255

                delta_ord_mag = max_magnitude_order - min_magnitude_order
                
                for rng in range(delta_ord_mag):
                    real_deltas = np.power(10., lndelta) * np.power(10., min_magnitude_order + rng)
                    for inner_rng in range(one_magnitude_range - 1):
                        img[((img > real_deltas[inner_rng]) & (img < real_deltas[inner_rng+1]))] = rgb_val
                        rgb_val += 1
                
                return img

            def _compute_halfgaussian_noise(data, plot=False):
                # fit dist to data
                mu, stdev = st.halfnorm.fit(data)

                print(f'\nMean and stdev of the half-gaussian that best fits with noise distribution:\nmu={mu}, stdev={stdev}')

                if plot:
                    plt.hist(data, bins = 40, range = (0., max(data)))
                    # Plot the PDF.
                    xmin, xmax = plt.xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = st.halfnorm.pdf(x, mu, stdev)
                    plt.plot(x, p, 'k', linewidth=2)
                return mu, stdev

            print('\nComputing max and min pixel value in order to scale image to RGB range')
            data_flat = self.original_cropped_image.flatten()
            neg_values = data_flat[data_flat < 0]
            max_val = max(data_flat)
            print(f'Max pixel value = {max_val}')

            # # histogram of noise (- noise, + noise)
            if plot_noise:
            # # we know that negative values are due to noise, and we assume a gaussian noise distribution
                min_val = min(data_flat)
                print(f'Min pixel value = {min_val}')
                # plt.hist(data_flat, bins = 40, range = (0.0001, 0.006))
                plt.hist(data_flat, bins = 40, range = (abs(min_val), max_val))
            
            print('Removing negative noise...')
            original_cropped_image_clipped = np.clip(self.original_cropped_image, a_min=0, a_max=np.max(self.image_data))
            print('Converting to RGB...')
            training_image, pixel_mean = _convert_to_RGB(original_cropped_image_clipped, np.abs(neg_values), max_val)
            
            return training_image, pixel_mean
        #####################


        # Downloading assets  
        for download_info in config.required_files:
            if not os.path.exists(os.path.join(config.TRAIN_DATA_FOLDER, download_info['file_name'])):
                utils.download_data(download_info['file_name'], download_info['url'], config.DOWNLOAD_FOLDER)

        self.raw_train_df = load_dataset()
        self.image_filename, self.image_data, self.image_header = load_train_image(print_info=print_info, use_pb=use_pb)
        self.wc = pywcs.WCS(self.image_header)
        self.image_width = self.image_header['NAXIS1']
        self.image_height = self.image_header['NAXIS2']
        self.pixel_res_x_arcsec = abs(float(self.image_header['CDELT1'])) * 3600
        self.pixel_res_y_arcsec = abs(float(self.image_header['CDELT2'])) * 3600
        self.BMAJ = float(self.image_header['BMAJ'])
        self.BMIN = float(self.image_header['BMIN'])
        self.psf_bmaj_ratio = (self.BMAJ / self.pixel_res_x_arcsec) * 3600.0
        self.psf_bmin_ratio = (self.BMIN / self.pixel_res_y_arcsec) * 3600.0
        self.g = 2 * self.pixel_res_x_arcsec # gridding kernel size as per specs
        self.g2 = self.g ** 2

        self.cleaned_train_df = process_dataset(use_pb, k)
        self.x1_min = int(np.floor(min(self.cleaned_train_df['x1'])))
        self.y1_min = int(np.floor(min(self.cleaned_train_df['y1'])))
        self.x2_max = int(np.floor(max(self.cleaned_train_df['x2'])))
        self.y2_max = int(np.floor(max(self.cleaned_train_df['y2'])))
        self.original_cropped_image = self.image_data[self.y1_min:self.y2_max, self.x1_min :self.x2_max]
        print()
        print('-'*10)
        print('Starting training image preprocessing...')
        self.training_image, self.pixel_mean = preprocess_train_image()
        print('End of training image preprocessing.')

        return

    def generate_patches(self, limit, plot_patches=False, source_dir=False):
        
        def _retrieve_patches_from_dir(limit, source_dir, plot_patches):

            patches_list = []
            proc_train_df = pd.DataFrame()
            previously_generate_patches = listdir(source_dir)
            patches_dict = {int(v.split('_')[0]):v for v in previously_generate_patches}

            if limit:
                print(f'Attention: by using limit parameter you will retrieve only the first {limit} patches')
            
            for patch_id in sorted(patches_dict)[:limit]:

                patch = patches_dict[patch_id]
                patches_list.append(patch)

                print(f'\nRetrieving patch {patch}')

                pkl = pd.read_pickle(f'{source_dir}/{patch}/{patch}.pkl')

                proc_train_df = proc_train_df.append(pkl)

                if plot_patches:
                    img_patch = np.load(f'{source_dir}/{patch}/{patch}.npy')
                    patch_xo = int(patch.split('_')[1])
                    patch_yo = int(patch.split('_')[2])
                    plt.imshow(img_patch, cmap='viridis', vmax=255, vmin=0)
                    print('Max gray level value = ', img_patch.max())
                    for box_index, box in pkl.iterrows():
                        # box = pkl.loc[pkl['ID']==box_index].squeeze()
                        plt.gca().add_patch(Rectangle((box.x1-patch_xo, box.y1-patch_yo), box.x2 - box.x1, box.y2-box.y1,linewidth=.5,edgecolor='r',facecolor='none'))
                        plt.text(box.x-patch_xo, box.y-patch_yo, box_index, fontsize=1)
                    plt.show()

            
            return patches_list, proc_train_df.reset_index()

        def _split_in_patch(patch_dim=100, is_multiple=False, show_plot=False, limit=None):
        
            def _cut_bbox(x, patch_xo, patch_yo, patch_dim):
                x.x1 = max(x.x1, patch_xo)
                x.y1 = max(x.y1, patch_yo)
                x.x2 = min(x.x2, patch_xo+patch_dim-1)
                x.y2 = min(x.y2, patch_yo+patch_dim-1)

                x = _from_image_to_patch_coord(x, patch_xo, patch_yo)

                return x

            def _from_image_to_patch_coord(x, patch_xo, patch_yo):

                x.x1s = x.x1 - patch_xo
                x.y1s = x.y1 - patch_yo
                x.x2s = x.x2 - patch_xo
                x.y2s = x.y2 - patch_yo

                return x

            def _save_bbox_files(img_patch, patch_id, df_scaled):
                
                if not os.path.exists(os.path.join(config.TRAIN_PATCHES_FOLDER, f"{patch_id}")):
                    os.makedirs(os.path.join(config.TRAIN_PATCHES_FOLDER, f"{patch_id}/"))

                np.save(os.path.join(config.TRAIN_PATCHES_FOLDER, f"{patch_id}/{patch_id}.npy"), img_patch)
                df_scaled.to_pickle(os.path.join(config.TRAIN_PATCHES_FOLDER, f"{patch_id}/{patch_id}.pkl"))
                print(f'Patch {patch_id} saved.')
                return
                    
            def _find_gt_in_patch(patch_xo, patch_yo, patch_dim, gt_df):

                def _filter_func(x, patch_xo=patch_xo, patch_yo=patch_yo, patch_dim=patch_dim):
                    return x.x >= patch_xo and x.y >= patch_yo and x.x <= patch_xo + patch_dim and x.y <= patch_yo + patch_dim and (x.x2 - x.x1) <= patch_dim * 0.8 and (x.y2 - x.y1) <= patch_dim * 0.8 
                    # return x.x >= patch_xo and x.y >= patch_yo and x.x <= patch_xo + patch_dim and x.y <= patch_yo + patch_dim

                filtered_df = gt_df[gt_df.apply(_filter_func, axis=1)]

                idx2 = filtered_df['ID'].tolist()
                return idx2

            h, w = self.training_image.shape
            fits_filename = self.image_filename.split('/')[-1].split('.')[0]
            hlim = round(np.sqrt(limit))
            wlim = hlim
            patches_without_gt = 0

            print(f'\nTraining image dimensions: {w} x {h}')
            print(f'Cutting training image in patches of dim {patch_dim}')
            proc_train_df = pd.DataFrame()
            # Add new columns to df
            self.cleaned_train_df['x1s'] = None
            self.cleaned_train_df['y1s'] = None
            self.cleaned_train_df['x2s'] = None
            self.cleaned_train_df['y2s'] = None
            self.cleaned_train_df['class_label'] = None
            self.cleaned_train_df['patch_id'] = None

            if (w % patch_dim !=0 or h % patch_dim != 0) and is_multiple:
                raise ValueError('Image size is not multiple of patch_dim. Please choose an appropriate value for patch_dim.')

            print(f'hlim: {hlim}')
            print(f'wlim: {wlim}')
            step = int(patch_dim/2)
            patches_list = []
            # for i in tqdm(range(0, h, int(patch_dim/2))):
            for i in tqdm(range(hlim)):
                    # for j in range(0, w, int(patch_dim/2)):
                    for j in range(wlim):
                            patch_xo = self.x1_min+j*step
                            patch_yo = self.y1_min+i*step
                            gt_id = []

                            gt_id = _find_gt_in_patch(patch_xo, patch_yo, patch_dim, self.cleaned_train_df)

                            # Questo potrebbe essere un buon punto in cui applicare il filtro sul noise

                            if len(gt_id) > 0:
                                print(f'\n Generating patch {len(patches_list)+1}/{limit}')

                                filename = f'{fits_filename}_{patch_xo}_{patch_yo}'
                                img_patch = self.training_image[i*step:i*step+patch_dim, j*step:j*step+patch_dim].copy()

                                # Cut bboxes that fall outside the patch
                                df_scaled = self.cleaned_train_df.loc[self.cleaned_train_df['ID'].isin(gt_id)].apply(_cut_bbox, patch_xo=patch_xo, patch_yo=patch_yo, patch_dim=patch_dim, axis=1)
                                df_scaled = df_scaled.loc[self.cleaned_train_df['ID'].isin(gt_id)].apply(_from_image_to_patch_coord, patch_xo=patch_xo, patch_yo=patch_yo, axis=1)
                                
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
                                df_scaled['class_label'] = df_scaled[['SIZE', 'CLASS']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1)
                                patch_index = i * (h // patch_dim) +j
                                patch_id = str(patch_index)+'_'+str(patch_xo)+'_'+str(patch_yo)+'_'+str(patch_dim)
                                df_scaled['patch_id'] = patch_id
                                
                                proc_train_df = proc_train_df.append(df_scaled)
                                _save_bbox_files(img_patch, patch_id, df_scaled)
                                patches_list.append(patch_id) 

                                if show_plot:
                                    plt.imshow(img_patch, cmap='viridis', vmax=255, vmin=0)
                                    print('Max gray level value = ', img_patch.max())
                                    for box_index in gt_id:
                                        box = df_scaled.loc[df_scaled['ID']==box_index].squeeze()
                                        plt.gca().add_patch(Rectangle((box.x1-patch_xo, box.y1-patch_yo), box.x2 - box.x1, box.y2-box.y1,linewidth=.5,edgecolor='r',facecolor='none'))
                                        plt.text(box.x-patch_xo, box.y-patch_yo, box_index, fontsize=1)
                                    plt.show()
                            else:
                                patches_without_gt +=1
                                pass

                                if len(patches_list) >= limit:
                                    break
                    else:
                        continue  # only executed if the inner loop did NOT break
                    break

            
            print(f'Patches with no gt: {patches_without_gt}')
            return patches_list, proc_train_df.reset_index()

        self.patch_list = {}
        
        if source_dir:
            self.patch_list, self.proc_train_df = _retrieve_patches_from_dir(limit, source_dir, plot_patches)
        else:
            self.patch_list, self.proc_train_df = _split_in_patch(config.patch_dim, show_plot=plot_patches, limit=limit)
        
        self.class_list = self.proc_train_df['class_label'].unique()
        print(f'Total number of generated patches: {len(self.patch_list)}')
        print()
        print(f'Class list: {self.class_list}')
        self.num_classes = len(self.proc_train_df['class_label'].unique())
        print(f'Number of distinct class labels: {self.num_classes}')
        return
    
    # TODO riscrivere usando la class list
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

        print('Number of possible class combinations:',self.num_combinations )
        #print('current combinations:',patch_class_int.unique() )
        # print(self.patch_list_per_class)    
        self.class_distribution = patch_class_int
        plt.hist(patch_class_int )
        #plt.legend(['columns >= 4 has 'f"{self.class_list[0]}",  'columns ... has'f"{self.class_list[1]}", 'odd columns has'f"{self.class_list[2]}" ])
        plt.xlabel("Classes and combinations")
        plt.ylabel("Num patches")
        plt.title("Histogram") #TODO: scriviamo label più esplicative
        plt.show()
        return

    def split_train_val_stratified(self, random_state, val_portion=.2):

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
        
        print('split ended')

        return

    # def split_train_val(self, random_state, val_portion=.2):
    #     # self.train_patch_list_no_strat = []
    #     # self.val_patch_list_no_strat = []
    #     train_class_distribution =[]

    #     train, val = train_test_split(self.patch_list,  test_size = val_portion, random_state=random_state)

    #     for idx in range(1, self.num_combinations+1):
    #         common_ID = set(train) & set(self.patch_list_per_class['key_{}'.format(idx)])

    #         if(len(self.patch_list_per_class['key_{}'.format(idx)])!=0):
    #             print("len of class:", format(idx), len(self.patch_list_per_class['key_{}'.format(idx)]))              
    #             print("len of common ID:", len(common_ID))
    #             train_class_distribution.append((len(common_ID)/len(self.patch_list_per_class['key_{}'.format(idx)]))*100)
    #         else:
    #             train_class_distribution.append(0)

    #     x_val= list(range(1, len(self.patch_list_per_class.keys())+1))
    #     # print("x", x_val)
    #     # print("y", train_class_distribution)
    #     # print("name",self.patch_list_per_class.keys() )
    #     plt.bar(x_val, train_class_distribution, tick_label =list(self.patch_list_per_class.keys()), width = 0.8)
    #     #plt.legend(['columns >= 4 has 'f"{self.class_list[0]}",  'columns ... has'f"{self.class_list[1]}", 'odd columns has'f"{self.class_list[2]}" ])
    #     plt.xlabel("classes ratio")
    #     plt.ylabel("num patches")
    #     plt.title("Histogram")
    #     plt.show() 

    def split_train_val(self, random_state, val_portion=.2):

        train, val = train_test_split(self.patch_list, test_size=val_portion, random_state=random_state)
        print(f'Train list consists of {len(train)} patches')
        print(f'Val list consists of {len(val)} patches')

        self.train_patch_list, self.val_patch_list = train, val

        self.plot_class_distribution([train, val])

        return

    def plot_class_distribution(self, l):

        if not isinstance(l, list):
            l = [l]

        fig, axes = plt.subplots(1,2, figsize=(10,5))
        # print(self.proc_train_df.shape)
        # display(self.proc_train_df)
        # ddf = self.proc_train_df['class_label'].value_counts()
        # display(ddf)
        # ddf.plot(kind='bar')

        for i, ll in enumerate(l):
            df = self.proc_train_df.loc[self.proc_train_df['patch_id'].isin(ll)]
            # print(df.shape)
            # display(df)
            # display(df['class_label'])
            # ddf = df['class_label'].value_counts()
            # display(ddf)
            df['class_label'].value_counts().plot(kind='bar', ax=axes[i])
            # df.hist(column='class_label', ax=axes[i])
            axes[i].set_xlabel("Class", labelpad=14)
            axes[i].set_ylabel("Frequency", labelpad=14)
            axes[i].set_title(f"Class distribution for {i}", y=1.02)
            
        plt.tight_layout()
        plt.show()

        return