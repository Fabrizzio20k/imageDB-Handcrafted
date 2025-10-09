import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List


class MetricsEvaluator:
    def __init__(self, retrieval_results_path: Path, queries_csv_path: Path):
        self.retrieval_df = pd.read_csv(retrieval_results_path)
        self.queries_df = pd.read_csv(queries_csv_path)
        self.k_values = [5, 10]

    def _calculate_ap(self, relevance: List[int]) -> float:
        if sum(relevance) == 0:
            return 0.0

        precisions = []
        num_relevant = 0

        for i, rel in enumerate(relevance, 1):
            if rel == 1:
                num_relevant += 1
                precision = num_relevant / i
                precisions.append(precision)

        return sum(precisions) / len(precisions) if precisions else 0.0

    def _extract_retrieved_classes(self, retrieval_row, max_items: int = 10) -> List[str]:
        retrieved_items = []

        for i in range(1, max_items + 1):
            col_name = f'retrieved_{i}'
            if col_name in retrieval_row and pd.notna(retrieval_row[col_name]):
                retrieved_path = retrieval_row[col_name]
                retrieved_class = Path(retrieved_path).parts[0]
                retrieved_items.append(retrieved_class)

        return retrieved_items

    def _calculate_relevance(self, retrieved_items: List[str], true_class: str) -> List[int]:
        return [1 if item == true_class else 0 for item in retrieved_items]

    def _calculate_precision_at_k(self, relevance: List[int], k: int) -> float:
        if k > len(relevance):
            return sum(relevance) / len(relevance) if relevance else 0.0
        return sum(relevance[:k]) / k

    def _calculate_majority_vote_accuracy(self, retrieved_items: List[str], true_class: str, top_n: int = 5) -> int:
        if len(retrieved_items) < top_n:
            return 0

        top_classes = retrieved_items[:top_n]
        majority_class = Counter(top_classes).most_common(1)[0][0]
        return 1 if majority_class == true_class else 0

    def _process_single_query(self, query_row) -> Dict:
        query_filename = query_row['filename']
        true_class = query_row['class_label']

        retrieval_row = self.retrieval_df[self.retrieval_df['query']
                                          == query_filename]

        if retrieval_row.empty:
            return None

        retrieval_row = retrieval_row.iloc[0]
        retrieved_items = self._extract_retrieved_classes(retrieval_row)

        if not retrieved_items:
            return None

        relevance = self._calculate_relevance(retrieved_items, true_class)

        query_metrics = {
            'query': query_filename,
            'true_class': true_class,
            'ap': self._calculate_ap(relevance)
        }

        for k in self.k_values:
            p_at_k = self._calculate_precision_at_k(relevance, k)
            query_metrics[f'p@{k}'] = p_at_k

        accuracy = self._calculate_majority_vote_accuracy(
            retrieved_items, true_class)
        query_metrics['accuracy'] = accuracy

        return query_metrics

    def calculate_metrics(self, output_folder: Path = None) -> Dict:
        if output_folder is None:
            output_folder = Path('Data/Results')

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        all_query_metrics = []
        query_ap_list = []
        query_p_at_k = {k: [] for k in self.k_values}
        query_accuracy_list = []

        for _, query_row in self.queries_df.iterrows():
            query_metrics = self._process_single_query(query_row)

            if query_metrics is None:
                continue

            all_query_metrics.append(query_metrics)
            query_ap_list.append(query_metrics['ap'])
            query_accuracy_list.append(query_metrics['accuracy'])

            for k in self.k_values:
                query_p_at_k[k].append(query_metrics[f'p@{k}'])

        overall_metrics = self._calculate_overall_metrics(
            query_ap_list, query_p_at_k, query_accuracy_list
        )

        self._save_results(all_query_metrics, overall_metrics, output_folder)

        return overall_metrics

    def _calculate_overall_metrics(self, query_ap_list: List[float],
                                   query_p_at_k: Dict[int, List[float]],
                                   query_accuracy_list: List[int]) -> Dict:
        metrics = {
            'mAP': sum(query_ap_list) / len(query_ap_list) if query_ap_list else 0.0,
            'accuracy': sum(query_accuracy_list) / len(query_accuracy_list) if query_accuracy_list else 0.0,
            'num_queries': len(query_ap_list)
        }

        for k in self.k_values:
            avg_p_at_k = sum(
                query_p_at_k[k]) / len(query_p_at_k[k]) if query_p_at_k[k] else 0.0
            metrics[f'P@{k}'] = avg_p_at_k

        return metrics

    def _save_results(self, all_query_metrics: List[Dict], overall_metrics: Dict, output_folder: Path):
        query_metrics_df = pd.DataFrame(all_query_metrics)
        query_metrics_file = output_folder / 'query_metrics.csv'
        query_metrics_df.to_csv(query_metrics_file, index=False)

        overall_metrics_df = pd.DataFrame([overall_metrics])
        overall_metrics_file = output_folder / 'overall_metrics.csv'
        overall_metrics_df.to_csv(overall_metrics_file, index=False)

        self._print_summary(
            overall_metrics, query_metrics_file, overall_metrics_file)

    def _print_summary(self, metrics: Dict, query_file: Path, overall_file: Path):
        print(f"\nMetrics Evaluation completed:")
        print(f"  - Number of queries: {metrics['num_queries']}")
        print(f"  - mAP: {metrics['mAP']:.4f}")

        for k in self.k_values:
            print(f"  - P@{k}: {metrics[f'P@{k}']:.4f}")

        print(f"  - Accuracy (majority vote): {metrics['accuracy']:.4f}")
        print(f"  - Query-level metrics saved to: {query_file}")
        print(f"  - Overall metrics saved to: {overall_file}")
