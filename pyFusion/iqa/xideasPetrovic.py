from abc import abstractmethod
import numpy as np
from math import pi
from cv2 import cartToPolar, Sobel, CV_32F
from matplotlib.pyplot import imshow, title, show

class XideaPetrovicMetric():
    # xydeas_petrovic parameters
    # The constants Γ, κ , σ  and Γα, κα, σα determine 
    # the  exact  shape  of  the  sigmoid  functions  used  to  form  the  edge  strength  and  
    # orientation  preservation  values.
    def __init__(self, image1, image2, fusedImage) -> None:
        self.EPS = np.finfo(float).eps
        self.GAMMA1 = 1
        self.GAMMA2 = 1
        self.K1 = -10
        self.K2 = -20
        self.DELTA1 = 0.5
        self.DELTA2 = 0.75
        self.L = 1
        self.image1 = image1
        self.image2 = image2
        self.fusedImage = fusedImage

    def _sobel_edge_detection(self, image, verbose=False):
        sx = Sobel(image, CV_32F, 1, 0)
        sy = Sobel(image, CV_32F, 0, 1)

        if verbose:
            imshow(sx, cmap='gray')
            title("Horizontal Edge")
            show()

            imshow(sy, cmap='gray')
            title("Vertical Edge")
            show()

        return cartToPolar(sx, sy)

    def _strenght_n_orientation(self, image):
        #The first input is the source image, which we convert to float. 
        #The second input is the output image, but we'll set that to None as we want the function 
        # call to return that for us. 
        #The third and fourth parameters specify the minimum and maximum values 
        # you want to appear in the output, which is 0 and 1 respectively, 
        #and the last output specifies how you want to normalize the image.
        # What I described falls under the NORM_MINMAX flag.
        #image = normalize(image.astype('float'), None, 0.0, 1.0, NORM_MINMAX)  
        # Kernels for convolving over the images
        #flt1= [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        #flt2= [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        # 1) get the map Sobel operator
        #fuseX = filter2D(image, -1, flt1)
        #fuseY = filter2D(image, -1, flt2)
        #   EQUIVALENT TO:
        s_x, s_y = self._sobel_edge_detection(image)
        #fusex
        # A Sobel edge operator is applied to yield the edge strength G
        g = np.sqrt(s_x**2 + s_y**2)
        # Orientation α(n,m) information for each pixel p
        alpha = np.arctan(s_y / ( s_x + self.EPS))
        return (g, alpha)

    def _perceptual_loss(self, gA, gF, alphaA, alphaF):
        # If g o alpha are followed by an underscore are ment to be considered written in uppercase
        # The relative strength and orientation values of g_AF(n,m) and alpha_AF(n,m) of an input 
        # image A with respect to F are formed as:
        
        #x, y = gA.shape
        #g_AF = np.zeros((x,y))
        #for n in range(x):
        #    for m in range(y):
        #        if (gA[n][m]  > gF[n][m]):
        #            g_AF[n][m] = gF[n][m] / ( gA[n][m] + EPS)
        #        else:
        #            g_AF[n][m] = gA[n][m] / ( gF[n][m] + EPS)
        bmap0 = gA > gF
        bmap1 = gA < gF
        
        g_AF0 = np.divide(gF, ( gA + self.EPS))
        g_AF1 = np.divide(gA, ( gF + self.EPS))

        g_AF = np.multiply(bmap0, g_AF0) + np.multiply(bmap1, g_AF1)

        alpha_AF = np.abs( np.abs(alphaA - alphaF) - pi/2) / (pi/2)

        qG_AF = self.GAMMA1 / (1 + np.exp( self.K1 *(g_AF - self.DELTA1)))
        qalpha_AF = self.GAMMA2 / (1 + np.exp( self.K2 *(alpha_AF - self.DELTA2) ))
        # These are used to derive the edge strength and orientation preservation values
        # QgAF(n,m)  and  QαAF(n,m)  model  perceptual  loss  of  information  in  F,  in  terms  of  
        # how well the strength and orientation values of a pixel p(n,m) in A are 
        # represented in the fused image. 
        #
        # Edge  information preservation values are then defined as
        q_AF = qG_AF * qalpha_AF
        # with  0  ≤  Q AF(n,m)  ≤  1 .  A  value  of  0  corresponds  to  the  complete  loss  of  edge  
        # information, at location (n,m), as transferred from A into F. QAF(n,m)=1 indicates 
        # “fusion” from A to F with no loss of information. 
        return q_AF

    def calculate(self):
        # EDGE Strenght and orientation for each pixels of the input images
        gA, alphaA = self._strenght_n_orientation(self.image1)
        gB, alphaB = self._strenght_n_orientation(self.image2)
        gF, alphaF = self._strenght_n_orientation(self.fusedImage)
        
        self.q_AF = self._perceptual_loss(gA, gF, alphaA, alphaF)
        self.q_BF = self._perceptual_loss(gB, gF, alphaB, alphaF)
        #
        # In general edge preservation values which 
        # correspond to pixels with high edge strength, should influence normalised weighted  
        # performance metric QP more than 
        # those of relatively low edge strength.Thus, wA(n,m)=[gA(n,m)]^L and 
        # wB(n,m)=[gB(n,m)]^L where L is a constant.
        #
        self.wA = gA #np.linalg.matrix_power(gA, L)
        self.wB = gB #np.linalg.matrix_power(gB, L)

        self.r = ( gF < gA ) | ( gF < gB )
        self.bitmap_artifacts = (gF > gA) & (gF > gB)

        return self.metric() #hook
    
    @abstractmethod
    def metric(self):
        pass

class InformationPreservation(XideaPetrovicMetric):
    def __init__(self, image1, image2, fusedImage) -> None:
        super().__init__(image1, image2, fusedImage)
    
    def metric(self):
        qP_ABF = sum( sum((self.q_AF * self.wA + self.q_BF * self.wB))) / sum ( sum((self.wA + self.wB)))
        return qP_ABF

class TotalFusionGain(XideaPetrovicMetric):
    def __init__(self, image1, image2, fusedImage) -> None:
        super().__init__(image1, image2, fusedImage)
    def metric(self):
        # local exclusive information in F, Q_delta
        # quantifies the total amount of local
        # exclusive information across the fused image.
    
        q_delta = np.abs(self.q_AF - self.q_BF)
    
        # For locations with strong correlation between the inputs Q_delta 
        # will be small or zero, indicating no exclusive
        # information. Conversely, in areas where one of the
        # inputs provides a meaningful feature that is not present
        # in the other this quantity will tend towards 1.

        # The common information component for all locations across the fused image
        q_common = (self.q_AF + self.q_BF  - q_delta) / 2
        # ½ is introduced as common information is contained in both Q_AF and Q_BF

        # Local estimates of exclusive information
        # components of each input
        q_delta_AF = self.q_AF - q_common
        # is the proportion of useful information fused in F that exists only in A
        q_delta_BF = self.q_BF - q_common
        # is the proportion of useful information fused in F that exists only in B

        # These quantities represent effectively, local fusion gain 
        # achieved by fusing A and B with respect to each individual {A, B}.
        
        # TOTAL FUSION GAIN
        return sum( sum((q_delta_AF * self.wA + q_delta_BF * self.wB))) / sum ( sum((self.wA + self.wB)))

class FusionLossArtifact(XideaPetrovicMetric):
    def __init__(self, image1, image2, fusedImage) -> None:
        super().__init__(image1, image2, fusedImage)
    
    def metric(self):
        # if gradient strength in F is larger than
        # that in the inputs, F contains artifacts; conversely, a
        # weaker gradient in F indicates a loss of input
        # information.
        loss = np.multiply(1 - self.q_AF, self.wA) + np.multiply(1 - self.q_BF, self.wB)
        loss = np.multiply(loss, self.r)
        loss = sum( sum(loss)) / sum ( sum((self.wA + self.wB)))

        #Artifacts calculation
        
        artifacts = 2 - self.q_AF - self.q_BF
        artifacts = np.multiply(self.bitmap_artifacts, artifacts)
        artifacts = sum( sum(artifacts)) / sum ( sum((self.wA + self.wB)))

        return (loss, artifacts)