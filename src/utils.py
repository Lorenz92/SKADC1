import pandas as pd
import numpy as np
import astropy.wcs as pywcs
import json
import os
import requests

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
