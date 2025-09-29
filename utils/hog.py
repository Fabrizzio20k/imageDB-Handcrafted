import numpy as np
import cv2


class HOG:
    def __init__(self, cell_size=8, block_size=2, bins=9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins

    def compute_gradients(self, image) -> tuple[np.ndarray, np.ndarray]:
        # PASO 1: Calcular gradientes
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image not found or unable to load.")

        # PASO 2: Calcular gradientes x e y usando Sobel
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

        # PASO 3: Calcular magnitud y ángulo
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * (180 / np.pi) % 180

        return magnitude, angle


if __name__ == "__main__":
    hog = HOG()
    hog.compute_gradients("caltech-101/airplanes/image_0001.jpg")
