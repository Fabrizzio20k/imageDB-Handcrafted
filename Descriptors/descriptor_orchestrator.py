import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from enum import Enum
from typing import List, Tuple, Optional

from .hog import HOG
from .lbp import LBP
from .color_hist import ColorHistogram
from .MixtureModel import RegionalMultiDescriptor


class DescriptorType(Enum):
    HOG = 'hog'
    LBP = 'lbp'
    COLOR = 'color'
    MIXTURE = 'mixture'


class DescriptorFactory:
    @staticmethod
    def create(descriptor_type: DescriptorType):
        descriptors = {
            DescriptorType.HOG: HOG(),
            DescriptorType.LBP: LBP(),
            DescriptorType.COLOR: ColorHistogram(),
            DescriptorType.MIXTURE: RegionalMultiDescriptor()
        }

        if descriptor_type not in descriptors:
            raise ValueError(f"Unsupported descriptor type: {descriptor_type}")

        return descriptors[descriptor_type]


class ImageLoader:
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')

    @staticmethod
    def collect_image_paths(base_path: str) -> List[Tuple[str, str, str]]:
        image_paths = []
        categories = [d for d in os.listdir(base_path)
                      if os.path.isdir(os.path.join(base_path, d))]

        for category in categories:
            category_path = os.path.join(base_path, category)
            for img_file in os.listdir(category_path):
                if img_file.lower().endswith(ImageLoader.VALID_EXTENSIONS):
                    img_path = os.path.join(category_path, img_file)
                    image_paths.append((img_path, category, img_file))

        return image_paths

    @staticmethod
    def load_image(img_path: str) -> Optional[np.ndarray]:
        image = cv2.imread(img_path)
        return image if image is not None else None


class FeatureExtractor:
    def __init__(self, descriptor_type: DescriptorType, descriptor):
        self.descriptor_type = descriptor_type
        self.descriptor = descriptor

    def extract_from_image(self, image: np.ndarray) -> np.ndarray:
        if self.descriptor_type == DescriptorType.HOG:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self.descriptor.compute_hog(gray)
        elif self.descriptor_type == DescriptorType.LBP:
            return self.descriptor.compute_histogram(
                self.descriptor.compute_lbp(image)
            )
        elif self.descriptor_type == DescriptorType.COLOR:
            return self.descriptor.compute_histogram(image)
        elif self.descriptor_type == DescriptorType.MIXTURE:
            return self.descriptor.extract(image)


class BatchProcessor:
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor

    def process_batch(self, img_paths_batch: List[Tuple[str, str, str]]):
        features_batch = []
        labels_batch = []
        filenames_batch = []

        for img_path, category, img_file in img_paths_batch:
            try:
                image = ImageLoader.load_image(img_path)
                if image is None:
                    continue

                features = self.feature_extractor.extract_from_image(image)

                features_batch.append(features)
                labels_batch.append(category)
                filenames_batch.append(f"{category}/{img_file}")
            except Exception:
                continue

        return features_batch, labels_batch, filenames_batch


class DescriptorOrchestrator:
    def __init__(self, descriptor_type: DescriptorType, caltech_path: str, output_path: str):
        self.descriptor_type = descriptor_type
        self.caltech_path = caltech_path
        self.output_path = output_path

        self.descriptor = DescriptorFactory.create(descriptor_type)
        self.feature_extractor = FeatureExtractor(
            descriptor_type, self.descriptor)
        self.batch_processor = BatchProcessor(self.feature_extractor)

    def _create_batches(self, items: List, batch_size: int) -> List[List]:
        return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

    def _process_all_batches(self, batches: List, n_workers: int):
        all_features = []
        all_labels = []
        all_filenames = []

        with Pool(processes=n_workers) as pool:
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for features_batch, labels_batch, filenames_batch in pool.imap(
                    self.batch_processor.process_batch, batches
                ):
                    all_features.extend(features_batch)
                    all_labels.extend(labels_batch)
                    all_filenames.extend(filenames_batch)
                    pbar.update(1)

        return np.array(all_features), all_labels, all_filenames

    def extract_features(self, batch_size: int = 32, n_workers: Optional[int] = None, force: bool = False):
        if n_workers is None:
            n_workers = min(cpu_count() - 1, 8)

        output_file = os.path.join(
            self.output_path, f'{self.descriptor_type.value}_features.pkl')

        if os.path.exists(output_file) and not force:
            print(f"\nFeatures file already exists: {output_file}")

            try:
                data = pd.read_pickle(output_file)
                if 'features' in data and 'labels' in data and 'filenames' in data:
                    print(f"  - Descriptor type: {self.descriptor_type.value}")
                    print(
                        f"  - Feature dimension: {data['features'].shape[1]}")
                    print(f"  - Total vectors: {data['features'].shape[0]}")
                    print("Use force=True to re-extract features.")
                    return output_file
                else:
                    print("  - File exists but format is invalid. Re-extracting...")
            except Exception as e:
                print(f"  - Error reading file: {e}. Re-extracting...")

        all_image_paths = ImageLoader.collect_image_paths(self.caltech_path)
        batches = self._create_batches(all_image_paths, batch_size)

        features_array, all_labels, all_filenames = self._process_all_batches(
            batches, n_workers
        )

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        data = {
            'features': features_array,
            'labels': all_labels,
            'filenames': all_filenames
        }

        pd.to_pickle(data, output_file)

        self._print_summary(features_array, output_file)

        return output_file

    def _print_summary(self, features_array: np.ndarray, output_file: str):
        print(f"\nFeature extraction completed:")
        print(f"  - Descriptor type: {self.descriptor_type.value}")
        print(f"  - Feature dimension: {features_array.shape[1]}")
        print(f"  - Total vectors generated: {features_array.shape[0]}")
        print(f"  - Saved to: {output_file}")
