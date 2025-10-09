import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


class MetricsEvaluator:
    def __init__(self, retrieval_results_path, queries_csv_path):
        self.retrieval_df = pd.read_csv(retrieval_results_path)
        self.queries_df = pd.read_csv(queries_csv_path)
        self.k_values = [5, 10]
        
    def calculate_metrics(self, output_folder='Data/Results'):
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        all_metrics = {}
        query_ap_list = []
        query_p_at_k = {k: [] for k in self.k_values}
        query_accuracy_list = []
        
        for _, query_row in self.queries_df.iterrows():
            query_filename = query_row['filename']
            true_class = query_row['class_label']
            
            retrieval_row = self.retrieval_df[self.retrieval_df['query'] == query_filename]
            
            if retrieval_row.empty:
                continue
            
            retrieval_row = retrieval_row.iloc[0]
            
            retrieved_items = []
            for i in range(1, 11):
                if f'retrieved_{i}' in retrieval_row:
                    retrieved_path = retrieval_row[f'retrieved_{i}']
                    if pd.notna(retrieved_path):
                        retrieved_class = retrieved_path.split('/')[0]
                        retrieved_items.append(retrieved_class)
            
            if not retrieved_items:
                continue
            
            relevance = [1 if item == true_class else 0 for item in retrieved_items]
            
            for k in self.k_values:
                p_at_k = sum(relevance[:k]) / k if k <= len(relevance) else sum(relevance) / len(relevance)
                query_p_at_k[k].append(p_at_k)
            
            ap = self._calculate_ap(relevance)
            query_ap_list.append(ap)
            
            if len(retrieved_items) >= 5:
                top_5_classes = retrieved_items[:5]
                majority_class =