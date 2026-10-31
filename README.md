# AuroraStay Policy Assistant (RAG POC)

An internal proof of concept for AuroraStay Hotels. Hotel staff type a question in plain English and get an answer grounded **only** in the hotel policy documents, with the supporting source chunks shown next to it.

## How it works
1. Policy PDFs are parsed, split into 120-token chunks (table rows keep their header row) and embedded with `HarishMaths/Hotel-Policy-Embedding`.
2. The 124 chunks are stored in a FAISS index (`backend/faiss_index/`).
3. A question is searched two ways, by meaning (vectors) and by exact words (BM25); the two rankings are merged and the top 8 chunks are sent to `gpt-4o-mini` with a prompt that allows answers only from those chunks.
4. A Flask API (port 7860) serves retrieval and answers; a Streamlit app (port 8501) is the user interface.

## Repository layout
```
backend/    Flask API: app.py, rag_utils.py, requirements.txt, Dockerfile, faiss_index/
frontend/   Streamlit UI: app.py, requirements.txt, Dockerfile
```

## Run in GitHub Codespaces
1. Add two Codespaces secrets for this repository: `OPENAI_API_KEY` and `OPENAI_API_BASE`.
2. Create a codespace on `main`, then build the images:
   ```bash
   cd backend  && docker build -t aurora-backend .  && cd ..
   cd frontend && docker build -t aurora-frontend . && cd ..
   ```
3. Start the backend (terminal 1), wait until the model has loaded, and make port 7860 public:
   ```bash
   docker run --name backend --network host -e OPENAI_API_KEY="$OPENAI_API_KEY" -e OPENAI_API_BASE="$OPENAI_API_BASE" aurora-backend
   ```
4. Start the frontend (terminal 2) and make port 8501 public:
   ```bash
   docker run --name frontend --network host aurora-frontend
   ```
5. Open the forwarded URL for port 8501.

## API
| Endpoint | Method | Input | Returns |
|---|---|---|---|
| `/` | GET | none | welcome message |
| `/v1/relevant_chunks` | POST | `query`, `k` (default 8) | retrieved chunks |
| `/v1/answer_with_relevant_chunks` | POST | `query`, `k`, `model_name`, `temperature`, `top_p`, `max_tokens` | answer and chunks |
| `/v1/answer` | POST | same as above | answer only |

## Notes
- No API keys are stored in this repository; they are read from environment variables.
- Results on a small benchmark are described in the project report; scores are indicative only (20 questions, LLM-judged).
