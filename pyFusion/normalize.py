import cv2   
    
def _RGB_to_YCbCr(selfimg_RGB):
    """
    A private method which converts an RGB image to YCrCb format
    """
    img_RGB = img_RGB.astype(np.float32) / 255.
    return cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

def _YCbCr_to_RGB(self, img_YCbCr):
    """
    A private method which converts a YCrCb image to RGB format
    """
    img_YCbCr = img_YCbCr.astype(np.float32)
    return cv2.cvtColor(img_YCbCr, cv2.COLOR_YCrCb2RGB)

def _is_gray(self, img):
    """
    A private method which returns True if image is gray, otherwise False
    """
    if len(img.shape) < 3:
        return True
    if img.shape[2] == 1:
        return True
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b == g).all() and (b == r).all():
        return True
    return False


def normalize(self):
    """
        Convert all images to YCbCr format
    """
    for idx, img in enumerate(self.input_images):
        if not self._is_gray(img):
            self.YCbCr_images[idx] = self._RGB_to_YCbCr(img)
            self.normalized_images[idx] = self.YCbCr_images[idx][:, :, 0]
        else:
            self.normalized_images[idx] = img / 255.