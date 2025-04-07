from database import detection, generate_test_data

print("Collection detection:", detection)
vectors, metadatas = generate_test_data()
print("Vectors:", vectors[0][:5], "...")
print("Metadatas:", metadatas[0])