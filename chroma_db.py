import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

def query_collection(input_text):
    if not input_text:
        return 0, ""

    df = pd.read_csv('embeddings.csv')
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
    chroma_client = chromadb.Client()

    client_persistent = chromadb.PersistentClient(path='/content/data_embeddings')
    db = client_persistent.create_collection(name='eamples_DB', embedding_function=sentence_transformer_ef)
    db.add(
        ids=[str(id) for id in df['ids'].tolist()],
        documents=df['examples'].tolist(),
        metadatas=df.drop(['ids', 'embeddings', 'examples'], axis=1).to_dict('records')
    )

    results = db.query(query_texts=[input_text], n_results=1)

    if results['distances'] and results['documents']:
        distance = results['distances'][0][0]
        documents = results['documents'][0][0]
    else:
        distance = 0
        documents = ""

    return distance, documents

# Texto de entrada para la consulta
input_text = "miercoles"

# Realizar la consulta y obtener el embedding más cercano y la distancia
distance, documents = query_collection(input_text)
print("Distancia:", distance)
print("Documentos:", documents)
