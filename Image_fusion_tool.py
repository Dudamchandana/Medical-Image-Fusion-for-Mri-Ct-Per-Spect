import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *
from random import *
from PIL import Image
from sklearn.decomposition import PCA
from pyFusion.legacy.basic_cv_tool import *

class Image_fusion_tool:
    
    def __init__(self, ImageName):
        self.ImageName = ImageName
    
    def weighted_average_fusion(self, im0, im1, w1, w2):
        '''
        This is the simplest image fusion algorithm. 
        :param im0: The first origin image.
        :param im1: The second origin image.
        :param w1: The weight of first image.
        :param w2: The weight of second image.
        :return: The fusioned image.
        '''
        if w1<0 or w2<0:
            print('invalid weight value')
            return
        elif w1 + w2 != 1:
            w1 = w1/(w1+w2)
            w2 = w2/(w1+w2)
        shape = np.shape(img1)
        img = np.zeros(shape,dtype = np.int8)
        if np.shape(im1) != shape:
            im1 = cv2.resize(img2, np.shape(im0), interpolation = cv2.INTER_CUBIC)
        img = w1*im0+w2*im1
        return img
    
    def PCA_image_fusion(self, im0, im1):
        '''
        This is the algorithm of image fusion based on PCA.
        :param img1: The origin image.
        :param img2: The high resolution image.
        :return: The fusioned image.
        '''
        estimator = PCA()
        estimator.fit(img1.copy())
        estimator.fit(img2.copy())
        img_f1 = estimator.transform(img1.copy())
        img_f2 = estimator.transform(img2.copy())
        img_f1[:,:40] = img_f2[:,:40]
        img = estimator.inverse_transform(img_f1)
        return img

    def PCA_fusion(self, im0, im1):
        #AUTOR: https://github.com/pfchai/ImageFusion/blob/master/src/main/fusion_pca.py
        imageSize = im0.size
        # Todo: for more than two images
        allImage = np.concatenate((im0.reshape(1, imageSize), im1.reshape(1, imageSize)), axis=0)
        covImage = np.cov(allImage)
        D, V = np.linalg.eig(covImage)
        if D[0] > D[1]:
            a = V[:,0] / V[:,0].sum()
        else:
            a = V[:,1] / V[:,1].sum()
        self._fusionImage = im0*a[0] + im1*a[1]
        return self._fusionImage
    
    def HSI_image_fusion(self, im0, im1):
        '''
        :param im0: The origin image.
        :param im1: The high resolution image.
        :return: The fusioned image.
        '''
        tool = basic_cv_tool('')
        hsi_im0 = tool.RGB2HSI(im0)
        hsi_im1 = tool.RGB2HSI(im1)
        hsi_im0[:,:,2] = hsi_im1[:,:,2]
        img = tool.HSI2RGB(hsi_im0)
        return img
    
    def gaussian_pyramid(self, img, level):
        temp = img.copy()
        pyramid_img = []
        for i in range(level):
            dst = cv2.pyrDown(temp)
            pyramid_img.append(dst)
            temp = dst.copy()
        return pyramid_img
    
    def laplacian_pyramid(self, img, level):
        pyramid_img = self.gaussian_pyramid(img, level)
        pyramid_lpls = []
        for i in range(level-1, -1, -1):
            if i-1<0:
                expend = cv2.pyrUp(pyramid_img[i], dstsize = img.shape[:2])
                lpls = cv2.subtract(img, expend)
                pyramid_lpls.append(lpls)
            else:
                expend = cv2.pyrUp(pyramid_img[i], dstsize = pyramid_img[i-1].shape[:2])
                lpls = cv2.subtract(pyramid_img[i-1], expend)
                pyramid_lpls.append(lpls)
        return pyramid_lpls
        
    def pyramid_image_fusion(self, im0, im1, fusion_rule, level = 2):
        #FIXME
        pass
        pyr_gimg1 = self.gaussian_pyramid(im0, level)
        pyr_gimg2 = self.gaussian_pyramid(im1, level)
        pyr_img1 = self.laplacian_pyramid(im0, level)
        pyr_img2 = self.laplacian_pyramid(im1, level)
        pyr_fusion = []
        for i in range(level):
            if fusion_rule == 'weighted':
                temp = self.weighted_average_fusion(pyr_img2[i], pyr_img1[i], 0.7, 0.3)
            elif fusion_rule == 'pca':
                temp = self.PCA_image_fusion(pyr_img2[i], pyr_img1[i])
            else :
                temp = self.HSI_image_fusion(pyr_img2[i], pyr_img1[i])

            pyr_fusion.append(temp)
        ls_ = pyr_gimg1[level-1]
        for i in np.arange(1,level,1):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, pyr_fusion[i-1])
        return temp
        
           
