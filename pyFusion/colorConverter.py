from cv2 import COLOR_RGB2YCrCb, COLOR_YCrCb2RGB, TonemapDrago, cvtColor, imread
from numpy import clip, float32, uint8 

class ColorConverter:

    def __init__(self, images_path) -> None:

        self.input_images = []
        self.normalized_images = [-1 for i in images_path]
        self.YCbCr_images = [-1 for i in images_path]
        for image in images_path:
            self.input_images.append( imread(image) )
        self._to_grayscale()

    def _RGB_to_YCbCr(self, img_RGB):
        """
        A private method which converts an RGB image to YCrCb format
        """
        img_RGB = img_RGB.astype(float32) / 255.
        return cvtColor(img_RGB, COLOR_RGB2YCrCb)

    def _YCbCr_to_RGB(self, img_YCbCr):
        """
        A private method which converts a YCrCb image to RGB format
        """
        img_YCbCr = img_YCbCr.astype(float32)
        return cvtColor(img_YCbCr, COLOR_YCrCb2RGB)

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

    def _to_grayscale(self):
        """
            A private method to convert all input images to YCbCr format
        """
        for idx, img in enumerate(self.input_images):
            if not self._is_gray(img):
                self.YCbCr_images[idx] = self._RGB_to_YCbCr(img)
                self.normalized_images[idx] = self.YCbCr_images[idx][:, :, 0]
            else:
                self.normalized_images[idx] = img / 255.

    def to_color(self, fused_img):
        # Reconstruct fused image given rgb input images
        for idx, img in enumerate(self.input_images):
            if not self._is_gray(img):
                self.YCbCr_images[idx][:, :, 0] = fused_img
                fused_final = self._YCbCr_to_RGB(self.YCbCr_images[idx])
                #
                # Given an interval, values outside the interval are clipped to the interval edges.
                # For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
                #
                fused_final = clip(fused_final, 0, 1)
        
        return (fused_final * 255).astype(uint8)
