import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.groq import Groq
import os
import chromadb
from dotenv import load_dotenv
load_dotenv()
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from rich import print,markdown

Settings.llm = Groq(model="openai/gpt-oss-120b",api_key =os.getenv("GROQ_API_KEY"))
# Sanity check: This should print your torch version without error now
print(f"Successfully loaded Torch: {torch.__version__}")


# 3. Data Ingestion with Section Metadata
def get_meta(file_path):
    # Extracts parent folder name (e.g., 'about', 'services') for GPS-like routing
    category = os.path.basename(os.path.dirname(file_path))
    url = f"https://ldce.ac.in/{category}/{os.path.basename(file_path)}".rstrip(".md")
    return {"category":category,"file_name": os.path.basename(file_path),"url":url}
# Point this to your uploaded folder of MD files
DATA_DIR = "root/"
reader = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True,file_metadata=get_meta)
documents = reader.load_data()
print(f"Loaded {len(documents)} documents from {DATA_DIR}")
for doc in documents:
    print(doc.metadata)
documents[:2]  # Display first 2 documents for verification

Settings.embed_model = HuggingFaceEmbedding(
    model_name="dunzhang/stella_en_1.5B_v5",
    device="cuda" if torch.cuda.is_available() else "cpu",
    truncate_dim = 1024,
    normalize=True,
)
print(f"Using embedding model on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("✅ Embedding model is ready!")


# 4. Parsing and Vector Storage
parser = MarkdownNodeParser(include_metadata=True,include_prev_next_rel=True,header_path_separator="/")
nodes = parser.get_nodes_from_documents(documents)
print(f"Parsed {len(nodes)} nodes from documents.")

db = chromadb.PersistentClient(path="./chroma_db3")
chroma_collection = db.get_or_create_collection("ld_chatbot_main")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# This generates embeddings and saves to the './chroma_db_colab' folder
index = VectorStoreIndex(nodes, storage_context=storage_context)
print(f"✅ Indexed {len(nodes)} chunks into ChromaDB on local.")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

retriever = .as_retriever()
print("✅ Retriever is set up and ready to go!")
retrieved = retriever.retrieve("what are the courses offered at LDCE?")
print(f"Retrieved {len(retrieved)} nodes for the query.")
for node in retrieved:  
    print(markdown.Markdown(f"**Source URL:** {node.metadata.get('url','N/A')}\n\n{node.get_text()}"))
    
  # Display first node for verification

