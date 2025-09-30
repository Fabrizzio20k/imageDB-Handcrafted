import numpy as np
import cv2


class HOG:
    def __init__(self, cell_size=8, block_size=2, bins=9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins

    def _compute_gradients(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = image.astype(np.float32)

        # PASO 1: Calcular gradientes x e y usando filtros de Sobel
        kernel_x = np.array([[-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1], [0], [1]], dtype=np.float32)

        gx = cv2.filter2D(image, cv2.CV_32F, kernel_x)
        gy = cv2.filter2D(image, cv2.CV_32F, kernel_y)

        # PASO 2: Calcular magnitud y ángulo
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * (180 / np.pi) % 180

        return magnitude, angle

    def _compute_cell_histograms(self, magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
        # PASO 3: Calcular histogramas por celda
        h, w = magnitude.shape
        n_cells_y = h // self.cell_size
        n_cells_x = w // self.cell_size

        histograms = np.zeros((n_cells_y, n_cells_x, self.bins))
        bin_width = 180 / self.bins

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_mag = magnitude[i*self.cell_size:(i+1)*self.cell_size,
                                     j*self.cell_size:(j+1)*self.cell_size]
                cell_ang = angle[i*self.cell_size:(i+1)*self.cell_size,
                                 j*self.cell_size:(j+1)*self.cell_size]

                for k in range(self.cell_size):
                    for l in range(self.cell_size):
                        mag = cell_mag[k, l]
                        ang = cell_ang[k, l]

                        # Interpolación bilineal entre bins
                        bin_center = ang / bin_width
                        bin_low = int(np.floor(bin_center)) % self.bins
                        bin_high = (bin_low + 1) % self.bins

                        # Peso proporcional a la distancia
                        weight_high = bin_center - np.floor(bin_center)
                        weight_low = 1 - weight_high

                        histograms[i, j, bin_low] += mag * weight_low
                        histograms[i, j, bin_high] += mag * weight_high

        return histograms

    def _normalize_blocks(self, histograms: np.ndarray) -> np.ndarray:
        # PASO 4: Normalizar bloques de celdas
        n_cells_y, n_cells_x, _ = histograms.shape
        block_h, block_w = self.block_size, self.block_size
        n_blocks_y = n_cells_y - block_h + 1
        n_blocks_x = n_cells_x - block_w + 1

        normalized_blocks = []

        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                block = histograms[i:i+block_h, j:j+block_w].flatten()
                norm = np.linalg.norm(block) + 1e-6  # Evitar división por cero
                normalized_block = block / norm
                normalized_blocks.append(normalized_block)

        return np.concatenate(normalized_blocks)

    def compute_hog(self, image_path: str | np.ndarray) -> np.ndarray:
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Image not found or unable to load.")
        else:
            image = image_path
            if len(image.shape) != 2:
                raise ValueError("Input image must be grayscale.")

        magnitude, angle = self._compute_gradients(image)
        histograms = self._compute_cell_histograms(magnitude, angle)
        hog_descriptor = self._normalize_blocks(histograms)

        return hog_descriptor

    def getDescriptorSize(self, img_h: int, img_w: int) -> int:
        n_cells_y = img_h // self.cell_size
        n_cells_x = img_w // self.cell_size
        n_blocks_y = n_cells_y - self.block_size + 1
        n_blocks_x = n_cells_x - self.block_size + 1
        block_vector_size = self.block_size * self.block_size * self.bins
        return n_blocks_y * n_blocks_x * block_vector_size
