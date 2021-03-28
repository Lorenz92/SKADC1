import pandas as pd
import numpy as np
import astropy.wcs as pywcs
import json


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_data(file, url, DOWNLOAD_FOLDER):
    data_path = os.path.join(DOWNLOAD_FOLDER, file)

    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    if os.path.exists(DOWNLOAD_FOLDER):
        print(f"Downloading SKA {file} data...")
        with requests.Session() as current_session:
            response = current_session.get(url, stream=True)
        save_response_content(response, data_path)
        print("Download completed!")

# BB plot test
def _get_bbox_from_ellipse(phi, r1, r2, cx, cy, h, w):
    """
    https://stackoverflow.com/questions/87734/
    how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse
    angle in degrees
    r1, r2 in number of pixels (half major/minor)
    cx and cy is pixel coordinated
    """
    half_pi = np.pi / 2
    ux = r1 * np.cos(phi)
    uy = r1 * np.sin(phi)
    vx = r2 * np.cos(phi + half_pi)
    vy = r2 * np.sin(phi + half_pi)

    hw = np.sqrt(ux * ux + vx * vx)
    hh = np.sqrt(uy * uy + vy * vy)
    x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh
    return (x1, y1, x2, y2)


def _gen_single_bbox(fits_fn, ra, dec, major, minor, pa):
    """
    Form the bbox BEFORE converting wcs to the pixel coordinates
    major and mior are in arcsec
    """
    ra = float(ra)
    dec = float(dec)
    fits_fn_dict = dict()

    if (fits_fn not in fits_fn_dict):
        hdulist = pyfits.open(fits_fn)
        height, width = hdulist[0].data.shape[0:2]
        w = pywcs.WCS(hdulist[0].header).deepcopy()
        fits_fn_dict[fits_fn] = (w, height, width)
    else:
        w, height, width = fits_fn_dict[fits_fn]

    cx, cy = w.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
    #cx = np.ceil(cx)
    if (not for_png):
        cx += 1
    #cy = np.ceil(cy)
    cy += 1
    if (cx < 0 or cx > width):
        print('got it cx {0}, {1}'.format(cx, fits_fn))
        return []
    if (cy < 0 or cy > height):
        print('got it cy {0}'.format(cy))
        return []
    if (for_png):
        cy = height - cy
    majorp = major / 3600.0 / pixel_res_x / 2 #actually semi-major 
    minorp = minor / 3600.0 / pixel_res_x / 2
    paa = np.radians(pa)
    x1, y1, x2, y2 = _get_bbox_from_ellipse(paa, majorp, minorp, cx, cy, height, width)
    # return x1, y1, x2, y2, height, width
    origin_area = (y2 - y1) * (x2 - x1)

    # crop it around the border
    xp_min = max(x1, 0)
    yp_min = max(y1, 0)
    xp_max = min(x2, width - 1)
    if (xp_max <= xp_min):
        return []
    yp_max = min(y2, height - 1)
    if (yp_max <= yp_min):
        return []
    new_area = (yp_max - yp_min) * (xp_max - xp_min)

    if (origin_area / new_area > 4):
        print('cropped box is too small, discarding...')
        return []
    return (xp_min, yp_min, xp_max, yp_max, height, width, cx, cy)

#TODO: move into dataset

def convert_boxes_to_px_coord(boxes_dataframe, fits_header):
    # convert all boxes into pix coords
    # boxes_dataframe = ska_dataset.raw_train_df
    fits_header = fits_header #data_560Mhz_1000h_fits[0].header
    image_width = fits_header['NAXIS1']
    image_height = fits_header['NAXIS2']
    pixel_res_x_arcsec = fits_header['CDELT1']
    pixel_res_y_arcsec = fits_header['CDELT2']

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

        x1, y1, x2, y2 = _get_bbox_from_ellipse(pa_in_rad, major_semia_px, minor_semia_px, cx, cy, image_height, image_width)

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

def extend_dataframe(df, cols_dict):
  # aggiungere x1... al df
  df_from_dict = pd.DataFrame.from_dict(cols_dict)
  if df.shape[0] != df_from_dict.shape[0]:
      raise ValueError("Dimension of DataFrame and dict passed don't match!")
  return pd.concat([df, df_from_dict], axis=1)



def split_in_patch(img, df, img_name, patch_dim=200):
    w, h = img.shape
    fits_filename = img_name.split('/')[-1].split('.')[0]
    
    images_dict = {
        "patch_name": [],
        "patch_xo": [],
        "patch_yo": [],
        "patch_dim": []
    }

    patches_json = {}

    # TODO: creare colonna vuota nel df per appartenenza riga a patch
    
    if w % patch_dim !=0 or h % patch_dim != 0:
        raise ValueError('Image size is not multiple of patch_dim. Please choose an appropriate value for patch_dim.')

    for i in range(0, w, patch_dim):
        for j in range(0, h, patch_dim):
            patch = {}
            img_patch = img[i:i+patch_dim, j:j+patch_dim]
            patch = {
                "orig_coords": img_patch.tolist(),
                # "scaled_coords": scaled_img_patch
            }

            # TODO: cerca le righe che appartengono alla patch e aggiorna nella nuova colonna
            
            filename = f'{fits_filename}_{i}_{j}'
            patches_json[filename] = patch

            images_dict["patch_name"].append(filename)
            images_dict["patch_xo"].append(i)
            images_dict["patch_yo"].append(j)
            images_dict["patch_dim"].append(patch_dim)

    with open(f'data/training/{fits_filename}.json', 'w', encoding='utf-8') as f:
        json.dump(patches_json, f, ensure_ascii=False, indent=4)

    
    return images_dict
