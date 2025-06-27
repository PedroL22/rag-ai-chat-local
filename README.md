# RAG AI Chat (Local)

A local Retrieval-Augmented Generation (RAG) chat application using LlamaIndex, Ollama, and Hugging Face embeddings. It indexes your transcripts and allows you to interactively query them with an LLM.

## Features

- Automatic GPU detection using PyTorch.
- Embedding generation with `BAAI/bge-small-en-v1.5` on GPU or CPU.
- Vector index persisted to `storage/`.
- Streaming query responses.
- Add new documents by dropping them into `transcripts/`.

## Requirements

- Python >= 3.9
- [Ollama](https://ollama.com)
- [PyTorch](https://pytorch.org) (CUDA support recommended)
- Python dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PedroL22/rag-ai-chat-local.git
   cd rag-ai-chat-local
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Ollama and pull the desired model:
   ```bash
   ollama pull llama3:8b
   ```

## Usage

1. Place your text transcripts in the `transcripts/` directory.
2. Run the application:
   ```bash
   python main.py
   ```
3. On first run, a vector index will be created in `storage/`. Subsequent runs will load the existing index.
4. Ask your questions when prompted. Type `exit` to quit.

## Configuration

Customize the following variables in `main.py`:

- `TEXTS_DIR`: Path to transcripts directory (default: `transcripts`).
- `INDEX_DIR`: Path to storage directory (default: `storage`).
- `OLLAMA_MODEL`: Ollama model to use (default: `llama3:8b`).

## Project Structure

```
rag-ai-chat-local/
├── main.py          # Entry point for the RAG chat application
├── requirements.txt # Python dependencies
├── transcripts/     # Directory containing text transcripts
└── storage/         # Persisted vector index files
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LlamaIndex](https://gpt-index.readthedocs.io/)
- [Ollama](https://ollama.com)
- [Hugging Face](https://huggingface.co)