import numpy as np
from scipy.ndimage import gaussian_filter

from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error

EPS = np.finfo(float).eps
#--------------------------
def ssim(origImage, fusedImage, mltchnl = False):
    return structural_similarity(origImage, fusedImage, data_range=fusedImage.max() - fusedImage.min(), multichannel = mltchnl)

def mse(origImage, fusedImage):
    return mean_squared_error(origImage, fusedImage)

#TODO: check https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/simple_metrics.py
def mutual_information_2d( origImage, fusedImage, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    origImage : 1D array
        first variable
    fusedImage : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    normalized : boolean
        default False
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(origImage.ravel(), fusedImage.ravel(), bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    gaussian_filter(jh, sigma=sigma, mode='constant',
                                    output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
                - np.sum(s2 * np.log(s2)))

    return mi

def entropy_2d(origImage,fusedImage):
    """
    Computes entropy between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    origImage : 1D array
        first variable
    fusedImage : 1D array
        second variable
    Returns
    -------
    entropy: float
        the computed entropy measure
    """
    # TODO: change bins to match variable img size
    bins = (256, 256)

    jh = np.histogram2d(origImage.ravel(), fusedImage.ravel(), bins=bins)[0]

    # compute marginal histograms
    jh = jh + EPS
    
    sh = np.sum(jh)
    pi = jh / sh
    entropy = -1 * np.sum(pi * np.log(pi))
    return entropy

def discrepancy(img1, img2):
    P, Q = img1.shape
    diff = img1 - img2
    d = np.sum(np.absolute(diff)) / P * Q

    return d