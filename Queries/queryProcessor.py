import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List
from Indexer.imageSimilarityIndex import ImageSimilarityIndex
from Descriptors.descriptor_orchestrator import DescriptorOrchestrator, DescriptorType


class QueryProcessor:
    VALID_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png')

    def __init__(self, index_path: Path, metadata_path: Path, descriptor_orchestrator: DescriptorOrchestrator, top_k: int = 10):
        self.indexer = ImageSimilarityIndex()
        self.indexer.load_index(index_path, metadata_path)
        self.descriptor_orchestrator = descriptor_orchestrator
        self.top_k = top_k

    def _collect_query_images(self, queries_path: Path) -> List[Path]:
        query_images = []
        for ext in self.VALID_EXTENSIONS:
            query_images.extend(queries_path.glob(ext))
        return query_images

    def _extract_features_from_image(self, image):
        descriptor_type = self.descriptor_orchestrator.descriptor_type

        if descriptor_type == DescriptorType.HOG:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self.descriptor_orchestrator.descriptor.compute_hog(gray)
        elif descriptor_type == DescriptorType.LBP:
            return self.descriptor_orchestrator.descriptor.compute_histogram(
                self.descriptor_orchestrator.descriptor.compute_lbp(image)
            )
        elif descriptor_type == DescriptorType.COLOR:
            return self.descriptor_orchestrator.descriptor.compute_histogram(image)
        elif descriptor_type == DescriptorType.MIXTURE:
            return self.descriptor_orchestrator.descriptor.extract(image)

    def _process_single_query(self, query_img_path: Path):
        image = cv2.imread(str(query_img_path))
        if image is None:
            return None

        features = self._extract_features_from_image(image)
        search_results = self.indexer.search(features, k=self.top_k)

        row_data = {'query': query_img_path.name}
        for i, result in enumerate(search_results, 1):
            row_data[f'retrieved_{i}'] = result['filename']
            row_data[f'score_{i}'] = result['score']

        return row_data

    def process_queries(self, queries_folder: Path, output_folder: Optional[Path] = None, grupo_nombre: str = "grupo"):
        if output_folder is None:
            output_folder = Path('Data/Results')

        queries_folder = Path(queries_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        query_images = self._collect_query_images(queries_folder)

        results_data = []

        for query_img_path in tqdm(query_images, desc="Processing queries"):
            row_data = self._process_single_query(query_img_path)
            if row_data is not None:
                results_data.append(row_data)

        results_df = pd.DataFrame(results_data)

        output_file = output_folder / f'results_{grupo_nombre}.csv'
        results_df.to_csv(output_file, index=False)

        self._print_summary(len(results_data), output_file)

        return output_file

    def _print_summary(self, num_queries: int, output_file: Path):
        print(f"\nQuery processing completed:")
        print(f"  - Number of queries processed: {num_queries}")
        print(f"  - Results saved to: {output_file}")
