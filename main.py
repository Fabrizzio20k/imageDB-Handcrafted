from core.ImageSimilarityIndex import ImageSimilarityIndex, IndexType, DescriptorType

dataset_dir = "caltech-101"
query_img = "caltech-101/airplanes/image_0001.jpg"

index = ImageSimilarityIndex(
    index_type=IndexType.HNSW, descriptor=DescriptorType.FISHER)
index.build_index(dataset_dir)

results = index.search(query_img, k=5)

print("Top imágenes similares a:", query_img)
for r in results[0]:
    print(r)
