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
        distance = results['distances'][0][0]
        documents = results['documents'][0][0]
    else:
        distance = 0
        documents = ""

    return distance, documents

# Texto de entrada para la consulta
input_text = "5"

# Realizar la consulta y obtener el embedding más cercano y la distancia
distance, documents = query_collection(input_text)
print(f"Distancia: {distance:.10f}")
print("Documentos:", documents)
