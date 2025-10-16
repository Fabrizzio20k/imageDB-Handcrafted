import numpy as np
import cv2


class ColorHistogram:
    def __init__(self, h_bins=8, s_bins=8, v_bins=8):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.feature_dim = h_bins + s_bins + v_bins
        
    def compute_histogram(self, image):
        if len(image.shape) == 2:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        hist_h = cv2.calcHist([hsv], [0], None, [self.h_bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [self.s_bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [self.v_bins], [0, 256])
        
        hist_h = hist_h.flatten()
        hist_s = hist_s.flatten()
        hist_v = hist_v.flatten()
        
        hist = np.concatenate([hist_h, hist_s, hist_v])
        
        hist = hist / (hist.sum() + 1e-7)
        
        return hist.astype(np.float32)
    
    def get_feature_dimension(self):
        return self.feature_dim