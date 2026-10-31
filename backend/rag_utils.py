
from langchain_openai import ChatOpenAI
import os

# ---------------------------------------------------------------------
# Hybrid retrieval = vector search + BM25 keyword search, fused with
# reciprocal rank fusion (RRF). The embedding model alone often ranks the
# right chunk low when the question uses everyday words or exact terms
# such as "Double Queen"; a keyword search catches those exact matches.
# ---------------------------------------------------------------------
import math, re, collections

_STOP = set("the a an and or of to in is are be for on at by with as it that this from not can may its if than then so do does you your i my me what how when who which will".split())

def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9\$]+", text.lower()) if w not in _STOP]

class BM25:
    """Small self-contained BM25 keyword index over LangChain documents."""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [collections.Counter(_tokens(d.page_content)) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(docs))
        df = collections.Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def top(self, query, n):
        q, scores = _tokens(query), []
        for i, t in enumerate(self.tf):
            s = 0.0
            for w in q:
                if w in t:
                    s += self.idf[w] * t[w] * (self.k1 + 1) / (t[w] + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.docs[i] for i in order[:n] if scores[i] > 0]

_bm25_cache = {}

def _get_bm25(vectorstore):
    # Built once per vector store from the chunks already stored in it
    if id(vectorstore) not in _bm25_cache:
        _bm25_cache[id(vectorstore)] = BM25(list(vectorstore.docstore._dict.values()))
    return _bm25_cache[id(vectorstore)]

# Retrieve the top-k relevant chunks (hybrid search)
def retrieve(query: str, k: int = 8, vectorstore=None) -> list[str]:
    pool, rrf_k = 20, 60
    vector_hits = vectorstore.similarity_search(query, k=pool)
    keyword_hits = _get_bm25(vectorstore).top(query, pool)

    # Reciprocal rank fusion: a chunk ranked high by either search rises to the top
    score, keep = {}, {}
    for ranking in (vector_hits, keyword_hits):
        for rank, d in enumerate(ranking):
            score[d.page_content] = score.get(d.page_content, 0) + 1 / (rrf_k + rank + 1)
            keep[d.page_content] = d
    top = [keep[c] for c in sorted(score, key=lambda c: -score[c])[:k]]

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in top
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about AuroraStay Hotels policies (booking, cancellation, rooms, refunds, charges, guest conduct).

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exclusions, limitations, exceptions, fees, deadlines, or property-specific details.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent fees, deadlines, limits, conditions, or property-specific details.
    10. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 8,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
