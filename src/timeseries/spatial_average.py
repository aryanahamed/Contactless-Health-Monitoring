import numpy as np
import cv2

def sa_region(patch):
    if patch is None:
        return np.array([np.nan, np.nan, np.nan]) # return nan if no patch

    # Apply gamma correction
    patch = gamma_correction(patch, gamma = 1.2)

    flat = patch.reshape(-1, 3) #flatning into 2d array ([b,g,r],3)
    valid = flat[np.any(flat > 0, axis=1)] #keeping only the non-black pixels

    # shouldnt happen but checking if entire mask is black
    if valid.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan])
    # grey scale intensity to filter very bright or dark pixels
    good_pixels = adaptive_intensity_filter(valid,lower_p=10, upper_p=90)

    if len(good_pixels) == 0:
        return np.array([np.nan, np.nan, np.nan]) #filtered everything out ig

    # filter 2: skin color check
    final_pixels = filter_skin_color(good_pixels)  # using default ycbcr ranges

    if final_pixels.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan])
    #using median instead of average incase-of outliers
    mean_bgr = np.mean(final_pixels, axis=0)
    return mean_bgr[::-1]  # BGR → RGB


def adaptive_intensity_filter(pixels, lower_p=10, upper_p=90):
    """
    keeps pixels that aren't offensively bright or dim, based on percentiles.
    """
    if pixels.shape[0] == 0:
        return pixels # nothing to see here, move along

    # cheap grayscale trick
    brightness = np.mean(pixels, axis=1)

    # figure out who's too dim or too bright for this party
    low_bar = np.percentile(brightness, min(lower_p, upper_p))
    high_bar = np.percentile(brightness, max(lower_p, upper_p))

    # edge case,if everyone's the same brightness
    if low_bar == high_bar:
        return pixels

    # the chosen ones
    within_range = (brightness >= low_bar) & (brightness <= high_bar)
    survivors = pixels[within_range]

    return survivors



def filter_skin_color(pixels, cr_min=120, cr_max=185, cb_min=60, cb_max=135):
    """
    skin color vibe check using ycbcr. keeps pixels that look kinda skin-like.
    input: (n, 3) numpy array, bgr format.
    """
    if pixels.shape[0] == 0:
        return pixels # can't filter ghosts

    #needs a specific shape, (n, 3) -> (n, 1, 3)
    pixels_to_convert = pixels.reshape(-1, 1, 3)
    ycbcr_pixels = cv2.cvtColor(pixels_to_convert, cv2.COLOR_BGR2YCrCb)
    # reshape back to (n, 3) because we like sanity
    ycbcr_pixels = ycbcr_pixels.reshape(-1, 3)

    # standard skin color ranges, allegedly. mostly care about cr/cb.
    cr_ok = (ycbcr_pixels[:, 1] >= cr_min) & (ycbcr_pixels[:, 1] <= cr_max)
    cb_ok = (ycbcr_pixels[:, 2] >= cb_min) & (ycbcr_pixels[:, 2] <= cb_max)
    skin_mask = cr_ok & cb_ok # gotta pass both

    # return the original bgr pixels that passed the skin check
    skin_pixels = pixels[skin_mask]
    return skin_pixels

def gamma_correction(img, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)
