import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from Indexer.image_similarity_index import ImageSimilarityIndex
from Descriptors.descriptor_orchestrator import DescriptorOrchestrator


class QueryProcessor:
    def __init__(self, index_path: str, metadata_path: str, descriptor_orchestrator: DescriptorOrchestrator, top_k: int = 10):
        self.indexer = ImageSimilarityIndex()
        self.indexer.load_index(index_path, metadata_path)
        self.descriptor_orchestrator: DescriptorOrchestrator = descriptor_orchestrator
        self.top_k = top_k

    def process_queries(self, queries_folder, output_folder='Data/Results'):
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        queries_path = Path(queries_folder)
        query_images = list(queries_path.glob('*.jpg')) + \
            list(queries_path.glob('*.jpeg')) + \
            list(queries_path.glob('*.png'))

        results_data = []

        for query_img_path in tqdm(query_images, desc="Processing queries"):
            query_name = query_img_path.name

            image = cv2.imread(str(query_img_path))
            if image is None:
                continue

            descriptor_type = self.descriptor_orchestrator.descriptor_type

            if descriptor_type == 'hog':
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                features = self.descriptor_orchestrator.descriptor.compute_hog(
                    gray)
            elif descriptor_type == 'lbp':
                features = self.descriptor_orchestrator.descriptor.compute_histogram(
                    self.descriptor_orchestrator.descriptor.compute_lbp(image)
                )
            elif descriptor_type == 'color':
                features = self.descriptor_orchestrator.descriptor.compute_histogram(
                    image)
            elif descriptor_type == 'mixture':
                features = self.descriptor_orchestrator.descriptor.extract(
                    image)

            search_results = self.indexer.search(features, k=self.top_k)

            row_data = {'query': query_name}
            for i, result in enumerate(search_results, 1):
                row_data[f'retrieved_{i}'] = result['filename']
                row_data[f'score_{i}'] = result['score']

            results_data.append(row_data)

        results_df = pd.DataFrame(results_data)

        output_file = os.path.join(output_folder, 'Retrieval_results.csv')
        results_df.to_csv(output_file, index=False)

        print(f"\nQuery processing completed:")
        print(f"  - Number of queries processed: {len(results_data)}")
        print(f"  - Results saved to: {output_file}")

        return output_file
