import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from .hog import HOG
from .lbp import LBP
from .color_hist import ColorHistogram
from .MixtureModel import RegionalMultiDescriptor
from pathlib import Path
import os

caltechpath = Path(os.path.join(Path.cwd(), 'Data',
                   'caltech-101', '101_ObjectCategories'))
outputpath = Path(os.path.join(Path.cwd(), 'Data',
                  'Feature_vectors_Caltech_101'))


class DescriptorOrchestrator:
    def __init__(self, descriptor_type='mixture'):
        self.descriptor_type = descriptor_type.lower()

        if self.descriptor_type == 'hog':
            self.descriptor = HOG()
        elif self.descriptor_type == 'lbp':
            self.descriptor = LBP()
        elif self.descriptor_type == 'color':
            self.descriptor = ColorHistogram()
        elif self.descriptor_type == 'mixture':
            self.descriptor = RegionalMultiDescriptor()
        else:
            raise ValueError(
                "Descriptor type must be 'hog', 'lbp', 'color', or 'mixture'")

    def _process_image_batch(self, img_paths_batch):
        features_batch = []
        labels_batch = []
        filenames_batch = []

        for img_path, category, img_file in img_paths_batch:
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue

                if self.descriptor_type == 'hog':
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    features = self.descriptor.compute_hog(gray)
                elif self.descriptor_type == 'lbp':
                    features = self.descriptor.compute_histogram(
                        self.descriptor.compute_lbp(image)
                    )
                elif self.descriptor_type == 'color':
                    features = self.descriptor.compute_histogram(image)
                elif self.descriptor_type == 'mixture':
                    features = self.descriptor.extract(image)

                features_batch.append(features)
                labels_batch.append(category)
                filenames_batch.append(f"{category}/{img_file}")
            except Exception:
                continue

        return features_batch, labels_batch, filenames_batch

    def extract_features(self, caltech_path=caltechpath, output_path=outputpath, batch_size=32, n_workers=None):
        Path(output_path).mkdir(parents=True, exist_ok=True)

        if n_workers is None:
            n_workers = min(cpu_count() - 1, 8)

        all_image_paths = []
        categories = [d for d in os.listdir(caltech_path)
                      if os.path.isdir(os.path.join(caltech_path, d))]

        for category in categories:
            category_path = os.path.join(caltech_path, category)
            for img_file in os.listdir(category_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_path, img_file)
                    all_image_paths.append((img_path, category, img_file))

        total_images = len(all_image_paths)

        batches = [all_image_paths[i:i+batch_size]
                   for i in range(0, total_images, batch_size)]

        all_features = []
        all_labels = []
        all_filenames = []

        with Pool(processes=n_workers) as pool:
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for features_batch, labels_batch, filenames_batch in pool.imap(
                    self._process_image_batch, batches
                ):
                    all_features.extend(features_batch)
                    all_labels.extend(labels_batch)
                    all_filenames.extend(filenames_batch)
                    pbar.update(1)

        features_array = np.array(all_features)

        df = pd.DataFrame(features_array)
        df['label'] = all_labels
        df['filename'] = all_filenames

        output_file = os.path.join(
            output_path, f'{self.descriptor_type}_features.csv')
        df.to_csv(output_file, index=False)

        print(f"\nFeature extraction completed:")
        print(f"  - Descriptor type: {self.descriptor_type}")
        print(f"  - Feature dimension: {features_array.shape[1]}")
        print(f"  - Total vectors generated: {features_array.shape[0]}")
        print(f"  - Saved to: {output_file}")

        return output_file


class FastDescriptorOrchestrator:
    def __init__(self, descriptor_type='mixture'):
        self.descriptor_type = descriptor_type.lower()

    def _extract_features_worker(self, args):
        img_path, category, img_file, descriptor_type = args

        try:
            image = cv2.imread(img_path)
            if image is None:
                return None

            if descriptor_type == 'hog':
                descriptor = HOG()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                features = descriptor.compute_hog(gray)
            elif descriptor_type == 'lbp':
                descriptor = LBP()
                features = descriptor.compute_histogram(
                    descriptor.compute_lbp(image)
                )
            elif descriptor_type == 'color':
                descriptor = ColorHistogram()
                features = descriptor.compute_histogram(image)
            elif descriptor_type == 'mixture':
                descriptor = RegionalMultiDescriptor()
                features = descriptor.extract(image)
            else:
                return None

            return features, category, f"{category}/{img_file}"
        except Exception:
            return None

    def extract_features(self, caltech_path=caltechpath, output_path=outputpath, n_workers=None):
        Path(output_path).mkdir(parents=True, exist_ok=True)

        if n_workers is None:
            n_workers = min(cpu_count() - 1, 12)

        all_tasks = []
        categories = [d for d in os.listdir(caltech_path)
                      if os.path.isdir(os.path.join(caltech_path, d))]

        for category in categories:
            category_path = os.path.join(caltech_path, category)
            for img_file in os.listdir(category_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_path, img_file)
                    all_tasks.append(
                        (img_path, category, img_file, self.descriptor_type))

        total_images = len(all_tasks)

        all_features = []
        all_labels = []
        all_filenames = []

        with Pool(processes=n_workers) as pool:
            with tqdm(total=total_images, desc="Extracting features") as pbar:
                for result in pool.imap_unordered(self._extract_features_worker, all_tasks, chunksize=10):
                    if result is not None:
                        features, category, filename = result
                        all_features.append(features)
                        all_labels.append(category)
                        all_filenames.append(filename)
                    pbar.update(1)

        features_array = np.array(all_features)

        df = pd.DataFrame(features_array)
        df['label'] = all_labels
        df['filename'] = all_filenames

        output_file = os.path.join(
            output_path, f'{self.descriptor_type}_features.csv')
        df.to_csv(output_file, index=False)

        print(f"\nFeature extraction completed:")
        print(f"  - Descriptor type: {self.descriptor_type}")
        print(f"  - Feature dimension: {features_array.shape[1]}")
        print(f"  - Total vectors generated: {features_array.shape[0]}")
        print(f"  - Saved to: {output_file}")

        return output_file


if __name__ == "__main__":
    orchestrator = FastDescriptorOrchestrator(descriptor_type='hog')
    orchestrator.extract_features(n_workers=8)
