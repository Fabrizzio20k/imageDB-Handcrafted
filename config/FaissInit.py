import faiss
import numpy as np


class FaissIndexFactory:
    def __init__(self, dim: int, index_type: str = "l2"):
        """
        Inicializa un índice FAISS con soporte para IDs.

        Args:
            dim (int): Dimensión de los vectores.
            index_type (str): Tipo de índice.
                              Opciones: "l2", "cosine", "hnsw", "lsh".
        """
        self.dim = dim
        self.index_type = index_type.lower()
        self.index = faiss.IndexIDMap(self._create_index())
        self.metadata = {}

    def _create_index(self):
        if self.index_type == "l2":
            return faiss.IndexFlatL2(self.dim)

        elif self.index_type == "cosine":
            return faiss.IndexFlatIP(self.dim)

        elif self.index_type == "hnsw":
            m = 32
            return faiss.IndexHNSWFlat(self.dim, m, faiss.METRIC_L2)

        elif self.index_type == "lsh":
            nbits = self.dim * 2
            return faiss.IndexLSH(self.dim, nbits)

        else:
            raise ValueError(f"Tipo de índice no soportado: {self.index_type}")

    def add(self, vectors: np.ndarray, labels: list):
        if self.index_type == "cosine":
            faiss.normalize_L2(vectors)

        ids = np.arange(len(self.metadata), len(self.metadata) + len(vectors))
        self.index.add_with_ids(vectors, ids)

        for i, meta in zip(ids, labels):
            self.metadata[i] = meta

    def search(self, queries: np.ndarray, k: int = 5):
        if self.index_type == "cosine":
            faiss.normalize_L2(queries)

        distances, indices = self.index.search(queries, k)

        results = []
        for row_ids, row_dists in zip(indices, distances):
            row = []
            for idx, dist in zip(row_ids, row_dists):
                if idx == -1:
                    continue
                meta = self.metadata[idx]
                row.append({
                    "id": int(idx),
                    "clase": meta["clase"],
                    "path": meta["path"],
                    "distancia": float(dist)
                })
            results.append(row)
        return results
