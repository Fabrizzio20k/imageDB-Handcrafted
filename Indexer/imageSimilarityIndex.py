import os
import time
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, List
from enum import Enum


class IndexType(Enum):
    L2 = 'l2'
    IP = 'ip'
    COSINE = 'cosine'
    HNSW = 'hnsw'
    LSH = 'lsh'


class ImageSimilarityIndex:
    def __init__(self, index_type: IndexType = IndexType.L2, nlist: int = 100, m: int = 32, nbits: int = 256):
        self.index_type = index_type
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.index = None
        self.filenames = []
        self.labels = []
        self.dimension = None

    def build_index(self, features_pkl_path: Path, output_path: Optional[Path] = None):
        if output_path is None:
            output_path = Path(features_pkl_path).parent / \
                'Indexed_feature_vectors'

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        data = pd.read_pickle(features_pkl_path)

        features = data['features'].astype(np.float32)
        self.labels = data['labels']
        self.filenames = data['filenames']

        self.dimension = features.shape[1]
        n_samples = features.shape[0]

        print(
            f"Building index with {n_samples} vectors of dimension {self.dimension}")

        self._create_index(features, n_samples)
        self._add_vectors_to_index(features)

        index_filename = output_path / f'{self.index_type.value}_index.faiss'
        metadata_filename = output_path / 'index_metadata.npz'

        faiss.write_index(self.index, str(index_filename))

        np.savez(str(metadata_filename),
                 filenames=self.filenames,
                 labels=self.labels,
                 index_type=self.index_type.value,
                 dimension=self.dimension)

        elapsed_time = time.time() - start_time

        self._print_summary(n_samples, elapsed_time,
                            index_filename, metadata_filename)

        return index_filename, metadata_filename

    def _create_index(self, features: np.ndarray, n_samples: int):
        if self.index_type == IndexType.L2:
            self.index = self._create_l2_index(features, n_samples)
        elif self.index_type == IndexType.IP:
            self.index = self._create_ip_index(features, n_samples)
        elif self.index_type == IndexType.COSINE:
            faiss.normalize_L2(features)
            self.index = self._create_ip_index(features, n_samples)
        elif self.index_type == IndexType.HNSW:
            self.index = self._create_hnsw_index()
        elif self.index_type == IndexType.LSH:
            self.index = self._create_lsh_index()
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def _create_l2_index(self, features: np.ndarray, n_samples: int):
        if n_samples > 10000:
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
            index.train(features)
            return index
        return faiss.IndexFlatL2(self.dimension)

    def _create_ip_index(self, features: np.ndarray, n_samples: int):
        if n_samples > 10000:
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(features)
            return index
        return faiss.IndexFlatIP(self.dimension)

    def _create_hnsw_index(self):
        index = faiss.IndexHNSWFlat(self.dimension, self.m)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 16
        return index

    def _create_lsh_index(self):
        return faiss.IndexLSH(self.dimension, self.nbits)

    def _add_vectors_to_index(self, features: np.ndarray):
        n_samples = features.shape[0]
        batch_size = 1000

        with tqdm(total=n_samples, desc="Adding vectors to index") as pbar:
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch = features[i:end_idx]

                if self.index_type == IndexType.COSINE:
                    faiss.normalize_L2(batch)

                self.index.add(batch)
                pbar.update(end_idx - i)

    def _print_summary(self, n_samples: int, elapsed_time: float, index_filename: Path, metadata_filename: Path):
        print(f"\nIndexing completed:")
        print(f"  - Index type: {self.index_type.value}")
        print(f"  - Number of vectors: {n_samples}")
        print(f"  - Dimension: {self.dimension}")
        print(f"  - Time taken: {elapsed_time:.2f} seconds")
        print(f"  - Index saved to: {index_filename}")
        print(f"  - Metadata saved to: {metadata_filename}")

    def load_index(self, index_path: Path, metadata_path: Path):
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        self.index = faiss.read_index(str(index_path))

        metadata = np.load(str(metadata_path), allow_pickle=True)
        self.filenames = metadata['filenames']
        self.labels = metadata['labels']

        index_type_str = str(metadata['index_type'])
        self.index_type = IndexType(index_type_str)
        self.dimension = int(metadata['dimension'])

    def search(self, query_features: np.ndarray, k: int = 10) -> List[Dict]:
        if self.index is None:
            raise ValueError("Index not built or loaded")

        query_features = query_features.astype(np.float32).reshape(1, -1)

        if self.index_type == IndexType.COSINE:
            faiss.normalize_L2(query_features)

        distances, indices = self.index.search(query_features, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append({
                    'filename': self.filenames[idx],
                    'label': self.labels[idx],
                    'distance': float(dist),
                    'score': float(1.0 / (1.0 + dist))
                })

        return results
