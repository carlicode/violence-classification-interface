import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

def query_collection(input_text):
    if not input_text:
        return 0, ""

    client_persistent = chromadb.PersistentClient(path='data_embeddings')
    db = client_persistent.get_collection(name='violence_embeddings_DB')
    results = db.query(query_texts=[input_text], n_results=1)

    if results['distances'] and results['documents']:
        documents = results['documents'][0][0]
        distance = results['distances'][0][0]
    else:
        documents = ""
        distance = 0

    return documents, distance