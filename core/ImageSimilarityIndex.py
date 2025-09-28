import os
import cv2
import numpy as np
from enum import Enum
from tqdm import tqdm
from config.FaissInit import FaissIndexFactory


class IndexType(Enum):
    L2 = "l2"
    COSINE = "cosine"
    HNSW = "hnsw"
    LSH = "lsh"


class ImageSimilarityIndex:
    def __init__(self, dim: int = 128, index_type: IndexType = IndexType.COSINE):
        self.index_type = index_type
        self.index = FaissIndexFactory(dim=dim, index_type=index_type.value)
        self.sift = cv2.SIFT_create()
        self.id_to_class = {}
        self.id_counter = 0

    def build_index(self, dataset_dir: str):
        all_files = []
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append((root, file))

        for root, file in tqdm(all_files, desc="Indexando imágenes", unit="img"):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, descriptors = self.sift.detectAndCompute(gray, None)
            if descriptors is not None:
                image_vector = np.mean(descriptors, axis=0).astype(
                    "float32").reshape(1, -1)
                class_label = os.path.basename(root)
                self.index.add(image_vector, [class_label])
                self.id_to_class[self.id_counter] = class_label
                self.id_counter += 1

    def search(self, query_path: str, k: int = 5):
        image = cv2.imread(query_path)
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None:
            return []
        query_vector = np.mean(descriptors, axis=0).astype(
            "float32").reshape(1, -1)
        return self.index.search(query_vector, k=k)
