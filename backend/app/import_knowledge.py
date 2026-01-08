import os
import sys
import glob
import libzim
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# Konfiguracja
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "knowledge"
KNOWLEDGE_DIR = "/app/knowledge"

def clean_html(html_content):
    # BeautifulSoup wymaga bytes lub str, nie memoryview
    soup = BeautifulSoup(html_content, "lxml")

    # Usuwamy zbędne elementy HTML
    for script in soup(["script", "style", "table", "footer", "nav", "aside", "header"]):
        script.extract()

    # POPRAWKA: Usunięto błędny parametr script=True, dodano strip=True dla czystszego tekstu
    text = soup.get_text(separator=" ", strip=True)
    return text

def import_zim_data():
    print(f"Connecting with Qdrant. Address: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)

    print("Loading embedding model (this might take a while)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    VECTOR_SIZE = 384

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists. Appending data.")

    zim_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.zim"))

    if not zim_files:
        print(f"Error: No .zim files found in directory '{KNOWLEDGE_DIR}'.")
        return

    print(f"Found {len(zim_files)} .zim file(s): {[os.path.basename(f) for f in zim_files]}")

    for zim_file_path in zim_files:
        print(f"\n--- Processing file: {zim_file_path} ---")
        try:
            archive = libzim.Archive(zim_file_path)
            total_entries = archive.all_entry_count
            print(f"Number of entries: {total_entries}")

            points_buffer = []
            processed_count = 0
            error_streak = 0

            for i in range(total_entries):
                try:
                    # Używamy _get_entry_by_id (zgodnie z nowym API libzim)
                    entry = archive._get_entry_by_id(i)

                    if entry.is_redirect:
                        continue

                    item = entry.get_item()

                    # Pobranie mimetype i bezpieczna konwersja
                    mimetype = getattr(item, 'mimetype', '')
                    if not isinstance(mimetype, str):
                        mimetype = str(mimetype)

                    valid_mime = mimetype in ["text/html", "application/xhtml+xml", "text/plain"]
                    valid_ext = entry.path.endswith(('.html', '.htm', '.txt'))

                    if not valid_mime and not valid_ext:
                        continue

                    # Konwersja memoryview na bytes
                    content_obj = item.content
                    if not content_obj:
                        continue

                    content_bytes = bytes(content_obj)

                    text = clean_html(content_bytes)

                    if len(text) < 100:
                        continue

                    chunks = [text[j:j+1000] for j in range(0, len(text), 1000)]

                    for chunk in chunks:
                        vector = model.encode(chunk).tolist()
                        points_buffer.append(PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={
                                "title": entry.title,
                                "text": chunk,
                                "source_file": os.path.basename(zim_file_path),
                                "path": entry.path,
                                "source": "zim_library"
                            }
                        ))

                    processed_count += 1
                    error_streak = 0

                    if len(points_buffer) >= 80:
                        client.upsert(collection_name=COLLECTION_NAME, points=points_buffer)
                        points_buffer = []
                        if processed_count % 500 == 0:
                            print(f"Analyzed articles: {processed_count}...", end='\r')

                except Exception as e:
                    error_streak += 1
                    if error_streak < 5:
                        print(f"Error processing entry {i}: {e}")
                    if error_streak > 200:
                        print("Too many errors in a row. Skipping rest of the file.")
                        break
                    continue

            if points_buffer:
                client.upsert(collection_name=COLLECTION_NAME, points=points_buffer)

            print(f"\nFinished file {zim_file_path}. Processed articles: {processed_count}")

        except Exception as e:
            print(f"Critical error processing file {zim_file_path}: {e}")
            continue

    print("\nAll files processed!")
