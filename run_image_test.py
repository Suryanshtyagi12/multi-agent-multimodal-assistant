from app.ingestion.image_ingest import ingest_image
from app.retrievers.image_retriever import retrieve_images

# STEP 1: Ingest an image (add any test image to your project)
ingest_image("D:\multi-agent-multimodel-assistant\Screenshot 2025-12-04 212903.png")   # replace with your image file

# STEP 2: Query by text
results = retrieve_images("login page error")
print(results)
