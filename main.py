import os

import torch  # Check for GPU availability using PyTorch
from llama_index.core import (Settings, SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex, load_index_from_storage)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# GPU CHECK
# Determine if CUDA GPU is available and set the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 50)
print(f"Detected compute device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
print("=" * 50)

# CONFIGURATION
TEXTS_DIR = "transcripts"
INDEX_DIR = "./storage"
OLLAMA_MODEL = "llama3:8b"

print("Starting RAG AI...")

# LOCAL MODEL SETUP
# 1. Initialize the LLM (uses GPU via Ollama)
llm = Ollama(model=OLLAMA_MODEL, request_timeout=120.0)

# 2. Configure the embedding model on the GPU
# Key change: added 'device=DEVICE'
print(f"Loading embedding model on device: {DEVICE}...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", device=DEVICE  # Key change
)
print("Embedding model loaded.")

# 3. Set default models for LlamaIndex
Settings.llm = llm
Settings.embed_model = embed_model

# Check if the index already exists
if not os.path.exists(INDEX_DIR):
    print(
        f"\nIndex not found at '{INDEX_DIR}'. Creating a new one using {DEVICE.upper()}..."
    )

    documents = SimpleDirectoryReader(TEXTS_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print("Index created and saved successfully!")
else:
    print(f"\nLoading existing index from '{INDEX_DIR}'...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
    print("Index loaded successfully!")

print("\nAI ready to receive questions. Type 'exit' to quit.")
print("-" * 50)
query_engine = index.as_query_engine(streaming=True)

# Interactive query loop
while True:
    pergunta = input("Your question: ")
    if pergunta.lower() == "exit":
        break

    response_stream = query_engine.query(pergunta)

    print("\nAI response:")
    response_stream.print_response_stream()
    print("\n" + "-" * 50)
