import os
import sys
import libzim
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "knowledge"
ZIM_FILE_PATH = "knowledge/knowledge_wikipedia.zim"

def clean_html(html_content):
    soup = BeautifulSoup(html_content, "lxml")
    for script in soup(["script", "style", "table", "footer", "nav"]):
        script.extract()

    text = soup.get_text(separator=" ", script=True)
    return text

def import_zim_data():
    # 1. Connection
    print(f"Connecting with Qdrant. Addres: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)
    print("Loading embedd model (this might take a while)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    VECTOR_SIZE = 384

    # 2. Creating collection
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    # 3. Reading text
    if not os.path.exists(ZIM_FILE_PATH):
        print(f"Error: {ZIM_FILE_PATH} not found.")
        return

    print(f"Opening archive: {ZIM_FILE_PATH}")
    archive = libzim.Archive(ZIM_FILE_PATH)

    total_entries = archive.all_entry_count
    print(f"Number of entries: {total_entries}")

    points_buffer = []
    processed_count = 0

    # 3. Iteration
    for i in range(total_entries):
        try:
            entry = archive.get_entry_by_index(i)


            if not entry.path.startswith('A/') or entry.is_redirect:
                continue

            item = entry.get_item()
            content_bytes = item.content

            title = entry.title


            if not content_bytes:
                continue

            text = clean_html(content_bytes)

            )
            if len(text) < 200:
                continue


            chunks = [text[j:j+1000] for j in range(0, len(text), 1000)]

            for chunk in chunks:
                vector = model.encode(chunk).tolist()

                points_buffer.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "title": title,
                        "text": chunk,
                        "source": "wikipedia"
                    }
                ))

            processed_count += 1

            if len(points_buffer) >= 100:
                client.upsert(collection_name=COLLECTION_NAME, points=points_buffer)
                points_buffer = []
                if processed_count % 100 == 0:
                    print(f"Analised articles: {processed_count}...", end='\r')

        except Exception as e:
            print(f"\nError {i}: {e}")
            continue

    # Wyślij resztki z bufora
    if points_buffer:
        client.upsert(collection_name=COLLECTION_NAME, points=points_buffer)

    print(f"\nDone! Number of processed articles: {processed_count}")

if __name__ == "__main__":
    import_zim_data()
