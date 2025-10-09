from pathlib import Path
from Descriptors.descriptor_orchestrator import DescriptorOrchestrator, DescriptorType
import os
from Indexer.imageSimilarityIndex import IndexType, ImageSimilarityIndex
from Queries.queryProcessor import QueryProcessor
from Queries.queryVisualizer import QueryVisualizer
from Metrics.metricsEvaluator import MetricsEvaluator


if __name__ == "__main__":
    GRUPO_NOMBRE = "grupo_05"

    caltech_path = Path('Data/caltech-101/101_ObjectCategories')
    features_output_path = Path('Data/Feature_vectors_Caltech_101')
    index_output_path = Path('Data/Indexed_feature_vectors')
    queries_folder = Path('Data/Queries')
    results_output_path = Path('Data/Results')

    descriptor_type = DescriptorType.MIXTURE
    index_type = IndexType.COSINE

    print("=" * 80)
    print("PASO 1: Extracción de características")
    print("=" * 80)

    orchestrator = DescriptorOrchestrator(
        descriptor_type=descriptor_type,
        caltech_path=str(caltech_path),
        output_path=str(features_output_path)
    )

    features_file = orchestrator.extract_features(
        batch_size=32,
        n_workers=os.cpu_count() - 5,
        force=False
    )

    print(f"\n✓ Características guardadas en: {features_file}")

    print("\n" + "=" * 80)
    print("PASO 2: Construcción del índice FAISS")
    print("=" * 80)

    indexer = ImageSimilarityIndex(index_type=index_type)
    index_file, metadata_file = indexer.build_index(
        features_pkl_path=Path(features_file),
        output_path=index_output_path
    )

    print(f"\n✓ Índice guardado en: {index_file}")
    print(f"✓ Metadatos guardados en: {metadata_file}")

    print("\n" + "=" * 80)
    print("PASO 3: Procesamiento de consultas")
    print("=" * 80)

    processor = QueryProcessor(
        index_path=index_file,
        metadata_path=metadata_file,
        descriptor_orchestrator=orchestrator,
        top_k=10
    )

    results_file = processor.process_queries(
        queries_folder=queries_folder,
        output_folder=results_output_path,
        grupo_nombre=GRUPO_NOMBRE
    )

    print(f"\n✓ Resultados guardados en: {results_file}")

    print("\n" + "=" * 80)
    print("PASO 4: Evaluación de métricas")
    print("=" * 80)

    queries_csv = queries_folder / 'queries.csv'

    if queries_csv.exists():
        evaluator = MetricsEvaluator(results_file, queries_csv)
        metrics = evaluator.calculate_metrics(results_output_path)

        print(f"\n{'MÉTRICAS DE EVALUACIÓN':^80}")
        print("=" * 80)
        print(f"  mAP:      {metrics['mAP']:.4f}")
        print(f"  P@5:      {metrics['P@5']:.4f}")
        print(f"  P@10:     {metrics['P@10']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print("=" * 80)

        composite_score = 0.5 * metrics['mAP'] + 0.3 * \
            metrics['P@10'] + 0.2 * metrics['accuracy']
        print(f"\n  Puntaje Compuesto (S): {composite_score:.4f}")
        print("=" * 80)
    else:
        print(
            f"\n⚠ Warning: {queries_csv} no encontrado. No se pueden calcular métricas.")

    print("\n" + "=" * 80)
    print("PASO 5: Visualización de resultados (opcional)")
    print("=" * 80)

    visualizer = QueryVisualizer(caltech_path, results_file)

    query_example = queries_folder / 'querie_01.jpg'
    if query_example.exists():
        print(f"\nVisualizando ejemplo: {query_example.name}")
        visualizer.visualize(query_example, k=10)
    else:
        print(
            f"\n⚠ Warning: {query_example} no encontrado para visualización.")

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO")
    print("=" * 80)
