import numpy as np


class LBP:
    def __init__(self, radius=1, points=8):
        self.radius = radius
        self.points = points
        self.uniform_patterns = self._get_uniform_patterns()
        
    def _get_uniform_patterns(self):
        uniform = {}
        label = 0
        for i in range(2**self.points):
            pattern = format(i, f'0{self.points}b')
            transitions = sum([pattern[j] != pattern[(j+1) % self.points] for j in range(self.points)])
            if transitions <= 2:
                uniform[i] = label
                label += 1
            else:
                uniform[i] = label
        return uniform
    
    def _get_pixel_neighbors(self, image, y, x):
        neighbors = []
        for p in range(self.points):
            angle = 2 * np.pi * p / self.points
            ny = y + self.radius * np.sin(angle)
            nx = x + self.radius * np.cos(angle)
            
            ny_floor, ny_ceil = int(np.floor(ny)), int(np.ceil(ny))
            nx_floor, nx_ceil = int(np.floor(nx)), int(np.ceil(nx))
            
            ny_floor = max(0, min(ny_floor, image.shape[0]-1))
            ny_ceil = max(0, min(ny_ceil, image.shape[0]-1))
            nx_floor = max(0, min(nx_floor, image.shape[1]-1))
            nx_ceil = max(0, min(nx_ceil, image.shape[1]-1))
            
            wy = ny - ny_floor
            wx = nx - nx_floor
            
            value = (1-wy) * (1-wx) * image[ny_floor, nx_floor] + \
                    wy * (1-wx) * image[ny_ceil, nx_floor] + \
                    (1-wy) * wx * image[ny_floor, nx_ceil] + \
                    wy * wx * image[ny_ceil, nx_ceil]
            
            neighbors.append(value)
        
        return neighbors
    
    def compute_lbp(self, image):
        if len(image.shape) == 3:
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        
        height, width = image.shape
        lbp_image = np.zeros((height, width), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                center = image[y, x]
                neighbors = self._get_pixel_neighbors(image, y, x)
                
                binary = 0
                for i, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        binary |= (1 << i)
                
                lbp_image[y, x] = self.uniform_patterns.get(binary, len(self.uniform_patterns)-1)
        
        return lbp_image
    
    def compute_histogram(self, lbp_image):
        n_bins = len(set(self.uniform_patterns.values()))
        hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-7)
        return hist