import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd


class QueryVisualizer:
    def __init__(self, caltech_path='Data/Caltech_101', results_csv='Data/Results/Retrieval_results.csv'):
        self.caltech_path = caltech_path
        self.results_df = pd.read_csv(results_csv)

    def visualize(self, query_image_path, k=10):
        query_name = os.path.basename(query_image_path)

        row = self.results_df[self.results_df['query'] == query_name]

        if row.empty:
            print(f"No results found for query: {query_name}")
            return

        row = row.iloc[0]

        fig, axes = plt.subplots(2, 6, figsize=(15, 6))
        axes = axes.flatten()

        query_img = cv2.imread(query_image_path)
        if query_img is not None:
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            axes[0].imshow(query_img)
            axes[0].set_title('Query Image', fontsize=10, fontweight='bold')
            axes[0].axis('off')

        axes[1].axis('off')

        for i in range(1, min(k+1, 11)):
            retrieved_col = f'retrieved_{i}'
            score_col = f'score_{i}'

            if retrieved_col not in row or pd.isna(row[retrieved_col]):
                axes[i+1].axis('off')
                continue

            retrieved_path = os.path.join(
                self.caltech_path, row[retrieved_col])
            retrieved_img = cv2.imread(retrieved_path)

            if retrieved_img is not None:
                retrieved_img = cv2.cvtColor(retrieved_img, cv2.COLOR_BGR2RGB)
                axes[i+1].imshow(retrieved_img)

                category = row[retrieved_col].split('/')[0]
                score = row[score_col]
                axes[i +
                     1].set_title(f'#{i}\n{category}\nScore: {score:.3f}', fontsize=9)
                axes[i+1].axis('off')
            else:
                axes[i+1].axis('off')

        plt.suptitle(
            f'Query Results for: {query_name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    visualizer = QueryVisualizer()
    visualizer.visualize('Data/Queries/Grupo5_queries/example.jpg')
