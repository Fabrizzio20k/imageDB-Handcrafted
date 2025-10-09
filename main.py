from pathlib import Path
from Descriptors.descriptor_orchestrator import DescriptorOrchestrator, DescriptorType
from Queries.queryProcessor import QueryProcessor
from Queries.queryVisualizer import QueryVisualizer
import os


if __name__ == "__main__":
    # Paso 1: Extraer características del descriptor
    caltech_path = Path(os.path.join(Path.cwd(), 'Data',
                        'caltech-101', '101_ObjectCategories'))
    output_path = Path(os.path.join(Path.cwd(), 'Data',
                       'Feature_vectors_Caltech_101'))

    orchestrator = DescriptorOrchestrator(
        descriptor_type=DescriptorType.HOG,
        caltech_path=str(caltech_path),
        output_path=str(output_path)
    )

    output_file = orchestrator.extract_features(batch_size=32, n_workers=12)

    print(f"Características guardadas en: {output_file}")

    # indexPath = Path.cwd() / 'Data' / 'Indexed_feature_vectors' / 'hnsw_index.faiss'
    # metadataPath = Path.cwd() / 'Data' / 'Indexed_feature_vectors' / 'index_metadata.npz'
    # processQueries = Path.cwd() / 'Data' / 'queries'

    # processor = QueryProcessor(
    #     index_path=str(indexPath),
    #     metadata_path=str(metadataPath),
    #     descriptor_orchestrator=descriptor,
    #     top_k=3
    # )
    # processor.process_queries(str(processQueries))

    # visualizer = QueryVisualizer(caltech_path=str(Path.cwd() / 'Data' / 'Caltech_101'),
    #                              results_csv=str(Path.cwd() / 'Data' / 'Results' / 'Retrieval_results.csv'))
    # processQueries = str(Path.cwd() / 'Data' / 'queries' / 'querie_01.jpg')
    # visualizer.visualize(processQueries, k=3)
