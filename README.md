# RAG System - Smart Multi-modal Information Handler

A powerful RAG (Retrieval-Augmented Generation) system for processing and querying documents, audio files, and multimedia content with support for local LLM models.

## Features

- **Multi-format Support**: PDF, DOCX, PPTX, TXT, and audio files (MP3, WAV, FLAC, etc.)
- **Local LLM Integration**: Support for Ollama, Hugging Face Transformers, and OpenAI
- **Vector Search**: ChromaDB for efficient similarity search
- **Audio Transcription**: Whisper integration for speech-to-text
- **Configurable**: Easy model switching and configuration
- **Interactive Mode**: Command-line interface for real-time queries

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up local LLM (optional)**:
   - For Ollama: Install [Ollama](https://ollama.ai/) and pull a model:
     ```bash
     ollama pull llama2
     ```
   - For Hugging Face: Models will be downloaded automatically on first use

## Quick Start

### 1. Basic Usage

```bash
# Embed documents
python main.py --embed document.pdf report.docx

# Query the system
python main.py --query "What is the main topic of the documents?"

# Interactive mode
python main.py --interactive
```

### 2. Configuration

The system uses `config.json` for configuration. Here's an example:

```json
{
  "llm": {
    "type": "ollama",
    "model_name": "llama2",
    "base_url": "http://localhost:11434"
  },
  "embedding": {
    "model": "all-mpnet-base-v2"
  },
  "audio": {
    "model": "base"
  },
  "database": {
    "path": "./chroma_db",
    "collection_name": "main_collection"
  },
  "processing": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_results": 5
  }
}
```

### 3. LLM Configuration Examples

#### Ollama (Recommended for local models)
```json
{
  "llm": {
    "type": "ollama",
    "model_name": "llama2",
    "base_url": "http://localhost:11434"
  }
}
```

#### Hugging Face Transformers
```json
{
  "llm": {
    "type": "transformers",
    "model_name": "microsoft/DialoGPT-medium",
    "device": "auto"
  }
}
```

#### OpenAI API
```json
{
  "llm": {
    "type": "openai",
    "model_name": "gpt-3.5-turbo",
    "api_key": "your-api-key-here"
  }
}
```

## Usage Examples

### Command Line Interface

```bash
# Show help
python main.py --help

# Embed multiple files
python main.py --embed file1.pdf file2.docx audio.mp3

# Single query
python main.py --query "Summarize the key points"

# Interactive mode
python main.py --interactive

# Show collection info
python main.py --info

# Clear collection
python main.py --clear
```

### Python API

```python
from modules.core import RAGCore
from modules import models

# Initialize with custom config
config = {
    "llm": {"type": "ollama", "model_name": "llama2"},
    "audio": {"model": "base"}
}

rag = RAGCore(llm_config=config["llm"], audio_model_name="base")

# Embed a file
result = rag.embed_file("document.pdf")
print(f"Embedded {result['chunks']} chunks")

# Query the system
response = rag.query("What is this document about?")
print(response['response'])

# Chat interface
messages = [
    {"role": "user", "content": "Hello, can you help me?"}
]
response = rag.chat(messages)
print(response)
```

## Supported File Types

### Text Documents
- **PDF**: `.pdf` - Text and images
- **Word**: `.docx` - Text and embedded images
- **PowerPoint**: `.pptx` - Text and images
- **Plain Text**: `.txt`

### Audio Files
- **MP3**: `.mp3`
- **WAV**: `.wav`
- **FLAC**: `.flac`
- **AAC**: `.aac`
- **OGG**: `.ogg`
- **M4A**: `.m4a`

## Architecture

```
RAG-System/
├── main.py                 # CLI interface
├── config.json            # Configuration
├── requirements.txt        # Dependencies
├── modules/
│   ├── core.py            # Main RAG core
│   ├── db.py              # ChromaDB operations
│   ├── embeddings.py      # Sentence transformers
│   ├── models.py          # LLM abstraction
│   └── parsing.py         # File content extraction
└── chroma_db/             # Vector database storage
```

## Advanced Configuration

### Custom Embedding Models
```python
# In embeddings.py, change the model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Smaller, faster model
```

### Custom Chunking Strategy
```python
# In core.py, modify _create_chunks method
def _create_chunks(self, text: str, chunk_size: int, chunk_overlap: int):
    # Custom chunking logic here
    pass
```

### Custom LLM Prompts
```python
# In core.py, modify _create_rag_prompt method
def _create_rag_prompt(self, question: str, context: str) -> str:
    return f"""You are a helpful assistant. Use the context to answer questions.

Context: {context}

Question: {question}

Answer:"""
```

## Troubleshooting

### Common Issues

1. **Ollama not running**:
   ```bash
   # Start Ollama service
   ollama serve
   ```

2. **CUDA out of memory**:
   - Use smaller models
   - Set `device: "cpu"` in config

3. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **File processing errors**:
   - Check file permissions
   - Ensure file is not corrupted
   - Verify supported file type

### Performance Tips

1. **Use smaller models** for faster processing
2. **Adjust chunk size** based on your content
3. **Use GPU** when available for transformers
4. **Batch process** multiple files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [OpenAI Whisper](https://github.com/openai/whisper) for audio transcription
- [Ollama](https://ollama.ai/) for local LLM support
