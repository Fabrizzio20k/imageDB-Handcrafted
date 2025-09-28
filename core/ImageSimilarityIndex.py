import os
import cv2
import numpy as np
from enum import Enum
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from config.FaissInit import FaissIndexFactory


class IndexType(Enum):
    L2 = "l2"
    COSINE = "cosine"
    HNSW = "hnsw"
    LSH = "lsh"


class DescriptorType(Enum):
    SIFT = "sift"
    HOG = "hog"
    LBP = "lbp"
    BOW = "bow"
    FISHER = "fisher"


class ImageSimilarityIndex:
    def __init__(self, descriptor: DescriptorType = DescriptorType.SIFT,
                 index_type: IndexType = IndexType.COSINE,
                 bow_k: int = 200):
        """
        Inicializa el índice con un descriptor elegido.

        Args:
            descriptor (DescriptorType): Tipo de descriptor a usar.
            index_type (IndexType): Tipo de índice FAISS.
            bow_k (int): Número de clusters para Bag of Words / Fisher.
        """
        self.descriptor_type = descriptor
        self.index_type = index_type
        self.bow_k = bow_k

        self.sift = cv2.SIFT_create() if descriptor in [DescriptorType.SIFT,
                                                        DescriptorType.BOW,
                                                        DescriptorType.FISHER] else None
        self.hog = cv2.HOGDescriptor() if descriptor == DescriptorType.HOG else None

        self.bow_model = None
        self.gmm = None
        self.train_descriptors = []

        dim = self._get_dim()
        self.index = FaissIndexFactory(dim=dim, index_type=index_type.value)

        self.id_counter = 0

    def _get_dim(self):
        """Retorna la dimensión del vector según el descriptor."""
        if self.descriptor_type == DescriptorType.SIFT:
            return 128
        elif self.descriptor_type == DescriptorType.HOG:
            return self.hog.getDescriptorSize()
        elif self.descriptor_type == DescriptorType.LBP:
            return 256
        elif self.descriptor_type == DescriptorType.BOW:
            return self.bow_k
        elif self.descriptor_type == DescriptorType.FISHER:
            return 2 * self.bow_k * 128
        else:
            raise ValueError("Descriptor no soportado")

    def _extract_features(self, image):
        """Extrae características según el descriptor elegido."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.descriptor_type == DescriptorType.SIFT:
            _, descriptors = self.sift.detectAndCompute(gray, None)
            return np.mean(descriptors, axis=0).astype("float32") if descriptors is not None else None

        elif self.descriptor_type == DescriptorType.HOG:
            gray = cv2.resize(gray, (64, 128))
            hog_vec = self.hog.compute(gray)
            return hog_vec.flatten().astype("float32")

        elif self.descriptor_type == DescriptorType.LBP:
            lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype("float32")
            hist /= (hist.sum() + 1e-6)
            return hist

        elif self.descriptor_type == DescriptorType.BOW:
            _, descriptors = self.sift.detectAndCompute(gray, None)
            if descriptors is None or self.bow_model is None:
                return None
            words = self.bow_model.predict(descriptors)
            hist, _ = np.histogram(
                words, bins=self.bow_k, range=(0, self.bow_k))
            hist = hist.astype("float32")
            hist /= (hist.sum() + 1e-6)
            return hist

        elif self.descriptor_type == DescriptorType.FISHER:
            _, descriptors = self.sift.detectAndCompute(gray, None)
            if descriptors is None or self.gmm is None:
                return None
            probs = self.gmm.predict_proba(descriptors)
            diff = descriptors[:, np.newaxis, :] - self.gmm.means_
            fisher_vec = np.concatenate([
                np.sum(probs[..., np.newaxis] * diff, axis=0).flatten(),
                np.sum(probs[..., np.newaxis] * (diff**2 -
                       self.gmm.covariances_), axis=0).flatten()
            ])
            fisher_vec = fisher_vec.astype("float32")
            return fisher_vec

    def build_index(self, dataset_dir: str):
        """Construye el índice a partir de un dataset de imágenes."""
        all_files = []
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append((root, file))

        if self.descriptor_type in [DescriptorType.BOW, DescriptorType.FISHER]:
            print("Recolectando descriptores para entrenamiento...")
            for root, file in tqdm(all_files, desc="Extrayendo descriptores", unit="img"):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, descriptors = self.sift.detectAndCompute(gray, None)
                if descriptors is not None:
                    self.train_descriptors.append(descriptors)

            if len(self.train_descriptors) > 0:
                all_desc = np.vstack(self.train_descriptors)
                if self.descriptor_type == DescriptorType.BOW:
                    self.bow_model = KMeans(
                        n_clusters=self.bow_k).fit(all_desc)
                elif self.descriptor_type == DescriptorType.FISHER:
                    self.gmm = GaussianMixture(
                        n_components=self.bow_k, covariance_type="diag").fit(all_desc)

        for root, file in tqdm(all_files, desc="Indexando imágenes", unit="img"):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            if image is None:
                continue

            vec = self._extract_features(image)
            if vec is not None:
                class_label = os.path.basename(root)
                self.index.add(vec.reshape(1, -1), [{
                    "clase": class_label,
                    "path": image_path
                }])
                self.id_counter += 1

    def search(self, query_path: str, k: int = 5):
        """Busca imágenes similares dado un query."""
        image = cv2.imread(query_path)
        if image is None:
            return []
        vec = self._extract_features(image)
        if vec is None:
            return []
        return self.index.search(vec.reshape(1, -1), k=k)
