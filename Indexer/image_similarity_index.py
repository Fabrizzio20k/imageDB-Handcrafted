import os
import time
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class ImageSimilarityIndex:
    def __init__(self, index_type='L2', nlist=100, m=32, nbits=256):
        self.index_type = index_type.lower()
        self.nlist = nlist
        self.m = m  # For HNSW
        self.nbits = nbits  # For LSH
        self.index = None
        self.filenames = []
        self.labels = []
        
    def build_index(self, features_csv_path, output_path='Data/Indexed_feature_vectors'):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        df = pd.read_csv(features_csv_path)
        
        self.labels = df['label'].values
        self.filenames = df['filename'].values
        
        feature_cols = [col for col in df.columns if col not in ['label', 'filename']]
        features = df[feature_cols].values.astype(np.float32)
        
        dimension = features.shape[1]
        n_samples = features.shape[0]
        
        print(f"Building index with {n_samples} vectors of dimension {dimension}")
        
        if self.index_type == 'l2':
            if n_samples > 10000:
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_L2)
                self.index.train(features)
            else:
                self.index = faiss.IndexFlatL2(dimension)
        
        elif self.index_type == 'ip':
            if n_samples > 10000:
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
                self.index.train(features)
            else:
                self.index = faiss.IndexFlatIP(dimension)
        
        elif self.index_type == 'cosine':
            faiss.normalize_L2(features)
            if n_samples > 10000:
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
                self.index.train(features)
            else:
                self.index = faiss.IndexFlatIP(dimension)
        
        elif self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(dimension, self.m)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
            
        elif self.index_type == 'lsh':
            self.index = faiss.IndexLSH(dimension, self.nbits)
            
        else:
            raise ValueError("index_type must be 'L2', 'IP', 'cosine', 'hnsw', or 'lsh'")
        
        with tqdm(total=n_samples, desc="Adding vectors to index") as pbar:
            batch_size = 1000
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch = features[i:end_idx]
                if self.index_type == 'cosine':
                    faiss.normalize_L2(batch)
                self.index.add(batch)
                pbar.update(end_idx - i)
        
        index_filename = os.path.join(output_path, f'{self.index_type}_index.faiss')
        faiss.write_index(self.index, index_filename)
        
        metadata_filename = os.path.join(output_path, 'index_metadata.npz')
        np.savez(metadata_filename, 
                 filenames=self.filenames, 
                 labels=self.labels,
                 index_type=self.index_type)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nIndexing completed:")
        print(f"  - Index type: {self.index_type}")
        print(f"  - Number of vectors: {n_samples}")
        print(f"  - Time taken: {elapsed_time:.2f} seconds")
        print(f"  - Index saved to: {index_filename}")
        print(f"  - Metadata saved to: {metadata_filename}")
        
        return index_filename, metadata_filename
    
    def load_index(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        metadata = np.load(metadata_path, allow_pickle=True)
        self.filenames = metadata['filenames']
        self.labels = metadata['labels']
        self.index_type = str(metadata['index_type'])
        
    def search(self, query_features, k=10):
        if self.index is None:
            raise ValueError("Index not built or loaded")
        
        query_features = query_features.astype(np.float32).reshape(1, -1)
        
        if self.index_type == 'cosine':
            faiss.normalize_L2(query_features)
        
        distances, indices = self.index.search(query_features, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:
                results.append({
                    'filename': self.filenames[idx],
                    'label': self.labels[idx],
                    'distance': dist,
                    'score': 1.0 / (1.0 + dist)
                })
        
        return results


if __name__ == "__main__":
    indexer = ImageSimilarityIndex(index_type='hnsw', m=32)
    indexer.build_index('Data/Feature_vectors_Caltech_101/mixture_features.csv')