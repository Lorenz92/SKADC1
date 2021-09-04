import os
from os import listdir
from IPython.core.display import display
import itertools
import numpy as np
import operator
import scipy.stats as st
from scipy.special import erfinv
import pandas as pd
import astropy.wcs as pywcs
from astropy.io import fits
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
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

    def __init__(self, k, print_info=False, show_plot= False, use_pb=False):
        
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
        self.stdev = None
        self.log_base = 10.


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

            def _dataset_plot():
                   # histogram of height and width
                plt.figure(figsize=(8,6))
                plt.hist(cleaned_df['width'], bins=400, alpha=0.5, label="width ")
                plt.hist(cleaned_df['height'], bins=200, alpha=0.5, label="height")
                plt.ylim((0, 11500))
                plt.xlim((0, 50))

                #scatter plot of class 
                patch_class_list = cleaned_df['class_label'].unique() 

                print(cleaned_df.groupby('class_label').count())
                print (patch_class_list)

                #fig = plt.figure(figsize=(8,8))
                #ax1 = fig.add_subplot(111)
                colors = cm.rainbow(np.linspace(0, 1, len(patch_class_list)))
                fig, axs = plt.subplots(1, len(patch_class_list), figsize=(4*len(patch_class_list),4))
                for idx, c in zip(range(len(patch_class_list)), colors):
                    #fig = plt.figure(figsize=(4,4))
                    clean_cass = cleaned_df.loc[(cleaned_df['class_label']== patch_class_list[idx]), :]
                    axs[idx].scatter(clean_cass['width'], clean_cass['height'], s=10,color=c, label=patch_class_list[idx])
                    axs[idx].set_xlim([0, 100])
                    axs[idx].set_ylim([0, 100])
                    axs[idx].title.set_text( f'width vs height of class: { patch_class_list[idx]}')
                    # plt.ylim((0, 100))
                    # plt.xlim((0, 100))
                    #plt.legend(loc='upper left')
                clean_cass2= cleaned_df.loc[(cleaned_df['class_label']== patch_class_list[4]), :]
                clean_cass1= cleaned_df.loc[(cleaned_df['class_label']== patch_class_list[3]), :]
                clean_cass3= cleaned_df.loc[(cleaned_df['class_label']== patch_class_list[2]), :]
                clean_cass4= cleaned_df.loc[(cleaned_df['class_label']== patch_class_list[1]), :]
                clean_cass5= cleaned_df.loc[(cleaned_df['class_label']== patch_class_list[0]), :]
                fig = plt.figure(figsize=(8,8))

                ax1 = fig.add_subplot(111)
                ax1.scatter(clean_cass2['width'], clean_cass2['height'], s=10, c='b', label=patch_class_list[4])
                ax1.scatter(clean_cass1['width'], clean_cass1['height'], s=10, c='r', label=patch_class_list[3])
                ax1.scatter(clean_cass3['width'], clean_cass3['height'], s=10, c='g', label=patch_class_list[2])
                ax1.scatter(clean_cass5['width'], clean_cass5['height'], s=10, c='c', label=patch_class_list[0])
                ax1.scatter(clean_cass4['width'], clean_cass4['height'], s=10, c='y', label=patch_class_list[1])
                plt.ylim((0, 150))
                plt.xlim((0, 250))
                plt.legend(loc='upper left')
                #colors = ['red','green','blue']          
                #ax1.scatter(clean_cass2['width'], clean_cass2['height'], c='b')# cmap=matplotlib.colors.ListedColormap(colors))
                
                #cb = plt.colorbar()
                # loc = np.arange(0,max(clean_cass2['class_label']),max(clean_cass2['class_label'])/float(len(colors)))
                # cb.set_ticks(loc)
                # cb.set_ticklabels(colors)
                return

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
            cleaned_df = copy.copy(boxes_dataframe.drop(index=id_to_delete).reset_index(drop=True))
            print(f'New dataset shape: {cleaned_df.shape}')
            print('Extending dataset with new computed columns...')
            cleaned_df = copy.copy(_extend_dataframe(cleaned_df, self.coords))
            print(f'Final cleaned dataset shape: {cleaned_df.shape}')
            print()
            cleaned_df['SIZE'] = cleaned_df['SIZE'].astype(int).astype('object')
            cleaned_df['CLASS'] = cleaned_df['CLASS'].astype(int).astype('object')
            cleaned_df['class_label'] = cleaned_df[['SIZE', 'CLASS']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1)
  
            if show_plot:
                _dataset_plot()

            if config.enlarge_bbox:
                print('Enlarging bboxes...')
                cleaned_train_df= copy.copy(cleaned_df.apply(_enlarge_bbox, scale_factor = config.bbox_scale_factor, axis=1))
                print('DONE - Enlarging bboxes...')
            cleaned_train_df = cleaned_train_df.astype({"ID": int})
            cleaned_train_df = cleaned_train_df.astype({"ID": object})
            display(cleaned_train_df.head())
            return cleaned_train_df.copy()
        
        
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
        self.training_image, self.pixel_mean, self.training_image_not_rgb, self.stdev = self.preprocess_train_image(plot_noise=show_plot)
        print('End of training image preprocessing.')

        return

    def preprocess_train_image(self, patch=None, plot_noise=False):
        """
        TODO: write desc
        """
        def _convert_to_RGB(image, data, max_val, patch_processing):
            print('Removing positive noise and rescaling to 0-255 interval...')
            if not patch_processing:
                mu, stdev = _compute_halfgaussian_noise(data, plot_noise)
            else:
                stdev = self.stdev
            
            thresh_low = stdev * 2.5
            min_magnitude_order = int(np.log10(thresh_low)) -1
            max_magnitude_order = int(np.log10(max_val))
            one_magnitude_range = int(np.floor(256/(max_magnitude_order - min_magnitude_order)))
            training_image = _RGB_to_FLOAT(image, one_magnitude_range, min_magnitude_order, max_magnitude_order)
            pixel_mean = np.repeat(int(np.mean(training_image)), 3)
            
            return training_image, pixel_mean, stdev

        def _RGB_to_FLOAT(image, one_magnitude_range, min_magnitude_order, max_magnitude_order):
            rgb_val = 0
            img=image.copy()
            thrsld_min = np.power(float(self.log_base), min_magnitude_order)
            thrsld_max = np.power(float(self.log_base), max_magnitude_order)
            img[img < thrsld_min] = rgb_val
            img[img > thrsld_max] = 255
            delta_ord_mag = max_magnitude_order - min_magnitude_order
            
            lndelta = np.linspace(1., float(self.log_base), one_magnitude_range)
            
            for rng in range(delta_ord_mag):
                real_deltas = (lndelta) * np.power(float(self.log_base), min_magnitude_order + rng)
            
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

        if patch is not None:
            img = patch
            plot_noise = False
            patch_processing = True
        else:
            img = self.original_cropped_image
            patch_processing = False

        print('\nComputing max and min pixel value in order to scale image to RGB range')
        data_flat = img.flatten()
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
        orig_img_zero_clipped = np.clip(img, a_min=0, a_max=np.max(self.image_data))
        print('Converting to RGB...')
        img_rgb, pixel_mean, stdev = _convert_to_RGB(orig_img_zero_clipped, np.abs(neg_values), max_val, patch_processing)
        
        return img_rgb, pixel_mean, orig_img_zero_clipped, stdev
    
    def generate_patches(self, limit, objects_to_ignore, plot_patches=False, source_dir=False, rgb_norm=False):
        def _extract_class_dict(df, filter, val, cols, key):
            return df[df[filter]==val][cols].set_index(key).T.to_dict('list')
        
        def _retrieve_patches_from_dir(limit, source_dir, plot_patches):

            patches_list = []
            patches_dict = {}
            proc_train_df = pd.DataFrame()
            previously_generate_patches = listdir(source_dir)
            patches_idx_dict = {int(v.split('_')[0]):v for v in previously_generate_patches}

            if limit:
                print(f'Attention: by using limit parameter you will retrieve only the first {limit} patches')
            
            for patch_id in sorted(patches_idx_dict)[:limit]:

                patch = patches_idx_dict[patch_id]
                patches_list.append(patch)
                
                print(f'\nRetrieving patch {patch}')

                pkl = pd.read_pickle(f'{source_dir}/{patch}/{patch}.pkl')
                patch_dict = _extract_class_dict(pkl,filter='patch_id', val=patch, cols=['class_label', 'patch_id'], key='class_label')
                patches_dict = utils.merge_dols(patches_dict, patch_dict)

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

            
            return patches_list, proc_train_df.reset_index(), patches_dict

        def _split_in_patch(patch_dim=100, is_multiple=False, show_plot=False, limit=None, objects_to_ignore=None, rgb_norm=False):
        
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

            def _check_overlap(obj_to_check, xo, yo, dim):
                x2 = xo + dim
                y2 = yo + dim
                overlap = False

                for _, row in obj_to_check.iterrows():
                    
                    if not (xo > row.x2 or x2 < row.x1 or yo > row.y2 or y2 < row.y1):
                        print(f'found overlap with {row.ID}!')
                        overlap = True 

                return overlap

            
            def _delete_crowded_objects(df):
                coords = np.array([df.x1s, df.y1s, df.x2s, df.y2s]).T
                area = np.array(df.width * df.height)

                _,_,res = utils.compute_intersection_fast(coords, area)

                return df.iloc[res].copy()
            
            h, w = self.training_image.shape
            fits_filename = self.image_filename.split('/')[-1].split('.')[0]
            hlim = round(np.sqrt(limit))
            wlim = hlim
            patches_without_gt = 0
            ignored_objects_patches = 0

            print(f'\nTraining image dimensions: {w} x {h}')
            print(f'Cutting training image in patches of dim {patch_dim}')
            proc_train_df = pd.DataFrame()

            # Add new columns to df
            self.cleaned_train_df['x1s'] = None
            self.cleaned_train_df['y1s'] = None
            self.cleaned_train_df['x2s'] = None
            self.cleaned_train_df['y2s'] = None
            self.cleaned_train_df['patch_id'] = None

            if (w % patch_dim !=0 or h % patch_dim != 0) and is_multiple:
                raise ValueError('Image size is not multiple of patch_dim. Please choose an appropriate value for patch_dim.')

            print(f'hlim: {hlim}')
            print(f'wlim: {wlim}')
            step = int(patch_dim/2)
            patches_list = []
            patches_dict = {}

            objects_to_ignore_df = self.cleaned_train_df.loc[self.cleaned_train_df['ID'].isin(objects_to_ignore)]
            objects_to_ignore_df = objects_to_ignore_df[['ID','x1', 'x2', 'y1', 'y2']]
            display(objects_to_ignore_df)

            # for i in tqdm(range(0, h, int(patch_dim/2))):
            for i in tqdm(range(hlim)):
                    # for j in range(0, w, int(patch_dim/2)):
                    for j in range(wlim):
                            patch_xo = self.x1_min+j*step
                            patch_yo = self.y1_min+i*step
                            gt_id = []

                            gt_id = _find_gt_in_patch(patch_xo, patch_yo, patch_dim, self.cleaned_train_df)

                            if np.isin(objects_to_ignore, gt_id).any():
                                print('Found objects to ignore.\nSkipping patch...')
                                continue
                            
                            # Check if patch contains residual of objects to ignore
                            if _check_overlap(objects_to_ignore_df, patch_xo, patch_yo, patch_dim):
                                ignored_objects_patches += 1
                                continue

                            if len(gt_id) > 0:
                                print(f'\n Generating patch {len(patches_list)+1}/{limit}')

                                filename = f'{fits_filename}_{patch_xo}_{patch_yo}'
                                if rgb_norm:
                                    img_patch = self.training_image_not_rgb[i*step:i*step+patch_dim, j*step:j*step+patch_dim].copy()
                                    img_patch, _, _, _ = self.preprocess_train_image(img_patch)
                                else:
                                    img_patch = self.training_image[i*step:i*step+patch_dim, j*step:j*step+patch_dim].copy()

                                # Cut bboxes that fall outside the patch
                                df_scaled = self.cleaned_train_df.loc[self.cleaned_train_df['ID'].isin(gt_id)].apply(_cut_bbox, patch_xo=patch_xo, patch_yo=patch_yo, patch_dim=patch_dim, axis=1)
                                df_scaled = df_scaled.loc[self.cleaned_train_df['ID'].isin(gt_id)].apply(_from_image_to_patch_coord, patch_xo=patch_xo, patch_yo=patch_yo, axis=1)
                                
                                df_scaled = _delete_crowded_objects(df_scaled)

                                # Here we need to update ground truth list based on last function called
                                gt_id = df_scaled['ID'].to_list()

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
                                patch_index = i * (h // patch_dim) +j
                                patch_id = str(patch_index)+'_'+str(patch_xo)+'_'+str(patch_yo)+'_'+str(patch_dim)
                                df_scaled['patch_id'] = patch_id
                                
                                proc_train_df = proc_train_df.append(df_scaled)
                                _save_bbox_files(img_patch, patch_id, df_scaled)
                                patches_list.append(patch_id)
                                patch_dict = _extract_class_dict(df_scaled,filter='patch_id', val=patch_id,cols=['class_label', 'patch_id'], key='class_label')
                                patches_dict = utils.merge_dols(patches_dict, patch_dict)
                                display(df_scaled)
                                if show_plot:
                                    plt.imshow(img_patch, cmap='viridis', vmax=255, vmin=0)
                                    print('Max gray level value = ', img_patch.max())
                                    for box_index in gt_id:
                                        box = df_scaled.loc[df_scaled['ID']==box_index].squeeze()
                                        plt.gca().add_patch(Rectangle((box.x1-patch_xo, box.y1-patch_yo), box.x2 - box.x1, box.y2-box.y1,linewidth=.5,edgecolor='r',facecolor='none'))
                                        plt.text(box.x-patch_xo, box.y-patch_yo, box_index, fontsize=5)
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
            print(f'Patches skipped due to ignore objects list: {ignored_objects_patches}')
            
            return patches_list, proc_train_df.reset_index(drop=True), patches_dict

        self.patch_list = []
        self.patches_dict = {}
        
        if source_dir:
            self.patch_list, self.proc_train_df, self.patches_dict = _retrieve_patches_from_dir(limit, source_dir, plot_patches)
        else:
            self.patch_list, self.proc_train_df, self.patches_dict = _split_in_patch(config.patch_dim, show_plot=plot_patches, limit=limit, objects_to_ignore=objects_to_ignore, rgb_norm=rgb_norm)
        
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

    def split_train_val(self, random_state, val_portion=.2, balanced=False):
        if not balanced:
            patch_list = self.patch_list
        else:
            self.balanced_patch_list = self.balance_patch_list()
            assert len(self.balanced_patch_list) > 1, 'Attention! There is no balanced_patch_list available!'
            patch_list = self.balanced_patch_list
            print(len(patch_list))

        train, val = train_test_split(patch_list, test_size=val_portion, random_state=random_state)
        print(f'Train list consists of {len(train)} patches')
        print(f'Val list consists of {len(val)} patches')

        self.train_patch_list, self.val_patch_list = train, val

        self.plot_class_distribution([train, val])

        return

    def plot_class_distribution(self, l):

        if not isinstance(l, list):
            l = [l]

        fig, axes = plt.subplots(1,2, figsize=(10,5))
        title_list = ['train set', 'val set']

        for i, ll in enumerate(l):
            class_count_dict = {k:0 for k in self.class_list}

            for p in ll:
                patch_class_count_dict = self.proc_train_df.loc[self.proc_train_df['patch_id']==p].value_counts(subset=['class_label'], sort=False).to_frame().reset_index().set_index('class_label').to_dict()[0]
                class_count_dict = utils.merge_dois(class_count_dict,patch_class_count_dict)
            
            axes[i].bar(range(len(class_count_dict)), list(class_count_dict.values()), align='center')
            
            axes[i].set_xticks(range(len(class_count_dict)))

            axes[i].set_xticklabels(list(class_count_dict.keys()))
            axes[i].set_xlabel("Class", labelpad=14)
            axes[i].set_ylabel("Frequency", labelpad=14)
            axes[i].set_title(f"Class distribution for {title_list[i]}", y=1.02)
            
        plt.tight_layout()
        plt.show()

        return

    def balance_patch_list(self):
        balanced_patch_list = self.patch_list.copy()
        class_freq_dict = {key: len(value) for key, value in self.patches_dict.items()}
        most_frequent_class = max(class_freq_dict.items(), key=operator.itemgetter(1))[0]
        max_freq = class_freq_dict[most_frequent_class]

        less_freq_classes = {k:v for k,v in class_freq_dict.items() if k != most_frequent_class}

        for key, _ in less_freq_classes.items():
            repeated_patches = []
            print('minor class:', self.patches_dict[key])
            patch_to_be_repeated = np.setdiff1d(self.patches_dict[key], self.patches_dict[most_frequent_class])
            print('patch_to_be_repeated:', patch_to_be_repeated)
            try:
                ratio = max_freq // len(patch_to_be_repeated)
                repeated_patches = np.tile(patch_to_be_repeated, ratio).tolist()
            except:
                continue
            balanced_patch_list += repeated_patches
        

        return balanced_patch_list