# Paul Graham Essays — RAG Application

A modular, CLI-based Retrieval-Augmented Generation (RAG) application built using Paul Graham's essays.

The application retrieves relevant passages from Paul Graham's essays using semantic search and uses Google's Gemini model to generate answers grounded in those passages.

This project is also a practical exploration of production-oriented AI engineering: ingestion pipelines, embedding models, vector databases, retrieval, LLM integration, evaluation, error handling, and testing.

## Features

- **Data ingestion:** Load Paul Graham's essays from Hugging Face.
- **Intelligent chunking:** Paragraph-aware, token-based text chunking with configurable overlap.
- **Pluggable embeddings:** Supports local BGE-M3 and OpenAI embeddings.
- **Vector search:** Store and retrieve essay chunks using Pinecone.
- **RAG generation:** Generate answers using Gemini with retrieved essay passages as context.
- **Source attribution:** Display the essay passages used to generate an answer.
- **Checkpointing:** Resume interrupted ingestion without unnecessarily reprocessing completed chunks.
- **Retry handling:** Retry transient failures using exponential backoff with jitter.
- **Evaluation groundwork:** Includes retrieval candidate collection and Recall@K, Precision@K, and MRR metric implementations. Manual relevance annotation and end-to-end evaluation are not yet exposed through the CLI.
- **CLI interface:** Run ingestion, retrieval, generation, and evaluation candidate collection from the terminal.

## Architecture

```text
                Hugging Face Dataset
                      |
                      v
                 Essay Loader
                      |
                      v
                 Text Chunker
                      |
                      v
                Embedding Model
                   (BGE-M3)
                      |
                      v
                  Pinecone
                      |
                      |
              User Question
                      |
                      v
              Query Embedding
                      |
                      v
             Pinecone Retrieval
                      |
                      v
              Retrieved Chunks
                      |
                      v
           Question + Retrieved Context
                      |
                      v
                 Gemini LLM
                      |
                      v
               Grounded Answer
                      |
                      v
               Source References
```

### RAG workflow

1. **Ingestion:** Essays are loaded from the Hugging Face dataset.
2. **Chunking:** Essays are divided into manageable, overlapping text chunks.
3. **Embedding:** Each chunk is converted into a vector representation.
4. **Indexing:** Vectors and associated metadata are stored in Pinecone.
5. **Retrieval:** A user query is embedded and used to retrieve relevant chunks.
6. **Generation:** Gemini receives the question and retrieved passages to generate an answer.
7. **Attribution:** Retrieved passages are displayed as sources.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.13+ |
| Package manager | uv |
| Source dataset | Hugging Face Datasets |
| Local embedding model | BAAI/bge-m3 |
| Alternative embedding model | OpenAI text-embedding-3-small |
| Vector database | Pinecone |
| LLM | Google Gemini |
| Tokenization | tiktoken |
| Configuration | python-dotenv |
| CLI | argparse |
| Testing | pytest |

## Project Structure

```text
pg-essays-agent/
├── src/
│   └── pg_essays_agent/
│       ├── checkpoint/
│       ├── chunking/
│       ├── embeddings/
│       ├── evaluation/
│       ├── generation/
│       ├── loaders/
│       ├── pipeline/
│       ├── retrieval/
│       ├── utils/
│       ├── vectorstores/
│       ├── application.py
│       ├── cli.py
│       ├── config.py
│       ├── exceptions.py
│       └── models.py
├── data/
│   └── evaluation/
├── tests/
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)
- A Pinecone account and API key
- A Google Gemini API key

For the default local embedding configuration, BGE-M3 model weights will be downloaded from Hugging Face when first required.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd pg-essays-agent
```

### 2. Install dependencies

```bash
uv sync --dev
```

### 3. Configure environment variables

Create your local environment file:

```bash
cp .env.example .env
```

Configure the required credentials in `.env`:

```dotenv
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key

# Embedding provider: local or openai
EMBED_PROVIDER=local
OPENAI_API_KEY=

# Hugging Face dataset
HF_DATASET=sgoel9/paul_graham_essays

# Pinecone configuration
PINECONE_INDEX=pg-essays-bge-m3
PG_NAMESPACE=essays
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Chunking
PG_CHUNK_TOKENS=400
PG_OVERLAP_TOKENS=60

# Ingestion
PG_BATCH=64

# Checkpoint
PG_CHECKPOINT=

# Gemini generation
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash

# Logging
LOG_LEVEL=INFO
```

The application selects the embedding model and default Pinecone index based on `EMBED_PROVIDER`.

| Provider | Model | Dimensions | Default index |
|---|---|---:|---|
| local | BAAI/bge-m3 | 1024 | pg-essays-bge-m3 |
| openai | text-embedding-3-small | 1536 | pg-essays-openai |

**Important:** The Pinecone index dimension must match the selected embedding model. Use the same embedding provider and index for ingestion and retrieval.

Never commit your `.env` file or expose API keys.

## Usage

The application exposes four CLI commands.

### 1. Ingest essays

Load the dataset, chunk the essays, generate embeddings, and store them in Pinecone.

```bash
uv run pg-essays-agent ingest
```

The ingestion pipeline uses checkpointing to resume previously completed work.

To clear the ingestion checkpoint and start ingestion again:

```bash
uv run pg-essays-agent ingest --fresh
```

### 2. Retrieve relevant essay passages

Search for relevant essay chunks without generating an LLM answer.

```bash
uv run pg-essays-agent query "What does Paul Graham mean by doing things that don't scale?"
```

Retrieve a different number of results:

```bash
uv run pg-essays-agent query "How should founders find startup ideas?" --top-k 10
```

The output includes:

- Essay title
- Similarity score
- Essay identifier
- Chunk index
- Retrieved text

### 3. Ask questions using RAG

Generate an answer using retrieved essay passages and Gemini.

```bash
uv run pg-essays-agent ask "What does Paul Graham mean by doing things that don't scale?"
```

Control the number of retrieved passages:

```bash
uv run pg-essays-agent ask "How do founders find startup ideas?" --top-k 5
```

The application retrieves relevant passages, supplies them as context to Gemini, and displays the generated answer along with its source references.

### 4. Collect retrieval evaluation candidates

Retrieve candidate chunks for the benchmark queries.

```bash
uv run pg-essays-agent collect-candidates \
    --input data/evaluation/benchmark.jsonl \
    --output data/evaluation/candidates.jsonl \
    --top-k 20
```

This produces a JSONL file containing candidate passages for relevance annotation and retrieval evaluation.

## Configuration Reference

| Variable | Description | Default |
|---|---|---|
| `PINECONE_API_KEY` | Pinecone API key | Required |
| `EMBED_PROVIDER` | Embedding provider (`local` or `openai`) | `local` |
| `OPENAI_API_KEY` | OpenAI API key when using OpenAI embeddings | Not set |
| `HF_DATASET` | Hugging Face dataset identifier | `sgoel9/paul_graham_essays` |
| `PINECONE_INDEX` | Pinecone index name | Provider-dependent |
| `PG_NAMESPACE` | Pinecone namespace | `essays` |
| `PINECONE_CLOUD` | Pinecone cloud provider | `aws` |
| `PINECONE_REGION` | Pinecone region | `us-east-1` |
| `PG_CHUNK_TOKENS` | Maximum chunk size in tokens | `400` |
| `PG_OVERLAP_TOKENS` | Chunk overlap in tokens | `60` |
| `PG_BATCH` | Ingestion batch size | `64` |
| `PG_CHECKPOINT` | Checkpoint file path | Provider-specific filename |
| `GEMINI_API_KEY` | Gemini API key | Required for `ask` |
| `GEMINI_MODEL` | Gemini model identifier | Application default |
| `LOG_LEVEL` | Logging level | `INFO` |
| `PG_MD_DIR` | Optional local Markdown source directory | Not set |

An optional local Markdown source directory can also be configured using `PG_MD_DIR`.

## Running Tests

Run the complete test suite:

```bash
uv run pytest -v
```

The tests cover application components including chunking, ingestion, checkpointing, embeddings, retrieval, evaluation metrics, retries, and CLI behavior.

## Current Scope

The initial release focuses on a command-line RAG application.

The project is being developed incrementally, with future improvements planned around:

- Retrieval quality evaluation and embedding comparisons.
- Reranking.
- RAG answer-quality evaluation.
- Observability and tracing.
- FastAPI service.
- Web interface and deployment.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Paul Graham](https://paulgraham.com/articles.html) — Original essays.
- [Hugging Face](https://huggingface.co/) — Dataset and model ecosystem.
- [Pinecone](https://www.pinecone.io/) — Vector database.
- [Google Gemini](https://ai.google.dev/) — LLM generation.
