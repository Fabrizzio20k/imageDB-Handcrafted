from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


class QueryVisualizer:
    def __init__(self, caltech_path: Path, results_csv: Path):
        self.caltech_path = Path(caltech_path)
        self.results_df = pd.read_csv(results_csv)

    def _load_and_prepare_image(self, image_path: Path, target_size=(200, 200)):
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img

    def _get_query_results(self, query_name: str):
        row = self.results_df[self.results_df['query'] == query_name]

        if row.empty:
            print(f"No results found for query: {query_name}")
            return None

        return row.iloc[0]

    def visualize(self, query_image_path: Path, k: int = 10, figsize: tuple = (16, 10)):
        query_image_path = Path(query_image_path)
        query_name = query_image_path.name

        row = self._get_query_results(query_name)
        if row is None:
            return

        k = min(k, 10)

        n_cols = 5
        n_rows = (k + n_cols) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [
            axes] if n_cols == 1 else axes

        query_img = self._load_and_prepare_image(
            query_image_path, target_size=(300, 300))
        if query_img is not None:
            axes[0].imshow(query_img)
            axes[0].set_title('QUERY IMAGE', fontsize=14, fontweight='bold',
                              color='darkred', pad=10)
            axes[0].axis('off')

            for spine in axes[0].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

        for i in range(1, n_cols):
            axes[i].axis('off')

        for i in range(1, k + 1):
            retrieved_col = f'retrieved_{i}'
            score_col = f'score_{i}'
            ax_idx = i + (n_cols - 1)

            if ax_idx >= len(axes):
                break

            if retrieved_col not in row or pd.isna(row[retrieved_col]):
                axes[ax_idx].axis('off')
                continue

            retrieved_path = self.caltech_path / row[retrieved_col]
            retrieved_img = self._load_and_prepare_image(retrieved_path)

            if retrieved_img is not None:
                axes[ax_idx].imshow(retrieved_img)

                category = Path(row[retrieved_col]).parts[0]
                score = row[score_col]

                title = f'Rank #{i}\n{category}\nScore: {score:.4f}'
                axes[ax_idx].set_title(title, fontsize=10, pad=8)
                axes[ax_idx].axis('off')

                color = 'green' if i <= 3 else 'blue' if i <= 6 else 'gray'
                for spine in axes[ax_idx].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
            else:
                axes[ax_idx].axis('off')

        for i in range(k + n_cols, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Query Results: {query_name}',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

    def visualize_multiple(self, query_image_paths: list, k: int = 5):
        for query_path in query_image_paths:
            self.visualize(query_path, k=k)
            print(f"\n{'='*80}\n")

    def save_visualization(self, query_image_path: Path, output_path: Path, k: int = 10):
        query_image_path = Path(query_image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        query_name = query_image_path.name
        row = self._get_query_results(query_name)
        if row is None:
            return

        k = min(k, 10)
        n_cols = 5
        n_rows = (k + n_cols) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
        axes = axes.flatten() if n_rows > 1 else [
            axes] if n_cols == 1 else axes

        query_img = self._load_and_prepare_image(
            query_image_path, target_size=(300, 300))
        if query_img is not None:
            axes[0].imshow(query_img)
            axes[0].set_title('QUERY IMAGE', fontsize=14, fontweight='bold',
                              color='darkred', pad=10)
            axes[0].axis('off')

        for i in range(1, n_cols):
            axes[i].axis('off')

        for i in range(1, k + 1):
            retrieved_col = f'retrieved_{i}'
            score_col = f'score_{i}'
            ax_idx = i + (n_cols - 1)

            if ax_idx >= len(axes):
                break

            if retrieved_col not in row or pd.isna(row[retrieved_col]):
                axes[ax_idx].axis('off')
                continue

            retrieved_path = self.caltech_path / row[retrieved_col]
            retrieved_img = self._load_and_prepare_image(retrieved_path)

            if retrieved_img is not None:
                axes[ax_idx].imshow(retrieved_img)
                category = Path(row[retrieved_col]).parts[0]
                score = row[score_col]
                title = f'Rank #{i}\n{category}\nScore: {score:.4f}'
                axes[ax_idx].set_title(title, fontsize=10, pad=8)
                axes[ax_idx].axis('off')
            else:
                axes[ax_idx].axis('off')

        for i in range(k + n_cols, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Query Results: {query_name}',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {output_path}")
