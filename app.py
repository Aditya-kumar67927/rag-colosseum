
import json
import re
from pathlib import Path

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fastembed import TextEmbedding

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

KB_FILE = BASE_DIR / "knowledge_base.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 3
MAX_TOP_K = 5

# Initial thresholds. These must be calibrated
# using the actual 20-question challenge.
ABSTAIN_THRESHOLD = 0.34
SENTENCE_THRESHOLD = 0.28

# Protect against simultaneous model access.



# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="RAG Colosseum API",
    description="Semantic retrieval and factual answering API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# REQUEST SCHEMA
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


# --------------------------------------------------
# GLOBAL VARIABLES
# --------------------------------------------------
model = None
passages = []
passage_embeddings = None
passage_ids = []
# --------------------------------------------------
# KNOWLEDGE BASE LOADER
# --------------------------------------------------

def load_knowledge_base():

    if not KB_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KB_FILE}"
        )

    with KB_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Support either a list or an object containing passages.
    if isinstance(data, dict):
        if "passages" in data:
            data = data["passages"]
        elif "knowledge_base" in data:
            data = data["knowledge_base"]
        else:
            raise ValueError(
                "JSON must contain a passages or knowledge_base list"
            )

    if not isinstance(data, list):
        raise ValueError("Knowledge base must be a JSON list")

    loaded_passages = []
    loaded_ids = []

    for index, item in enumerate(data):

        if isinstance(item, str):
            text = item

        elif isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("content")
                or item.get("passage")
            )

        else:
            text = None

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Missing passage text at index {index}"
            )

        loaded_passages.append(text.strip())
        loaded_ids.append(index)

    if len(loaded_passages) != 50:
        raise ValueError(
            f"Expected exactly 50 passages, "
            f"found {len(loaded_passages)}"
        )

    return loaded_passages, loaded_ids


# --------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------


def encode_texts(texts):
    """Generate normalized embeddings using FastEmbed."""
    vectors = np.asarray(
        list(model.embed(texts, batch_size=16)),
        dtype=np.float32
    )

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


@app.on_event("startup")
def initialize():
    global model
    global passages
    global passage_embeddings
    global passage_ids

    print("Loading knowledge base...")

    passages, passage_ids = load_knowledge_base()

    print("Loaded passages:", len(passages))

    print("Loading FastEmbed ONNX model...")

    model = TextEmbedding(
        model_name=MODEL_NAME,
        threads=1
    )

    print("Generating passage embeddings...")

    passage_embeddings = encode_texts(passages)

    print("RAG system ready!")


# --------------------------------------------------
# PROMPT INJECTION FILTER
# --------------------------------------------------

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"reveal (the )?(system )?prompt",
    r"disregard (all )?instructions",
    r"you are now",
    r"system prompt:",
    r"developer message:",
    r"send .*password",
    r"exfiltrate"
]


def is_suspicious_sentence(sentence):

    text = sentence.lower()

    return any(
        re.search(pattern, text)
        for pattern in INJECTION_PATTERNS
    )


# --------------------------------------------------
# SEMANTIC RETRIEVAL
# --------------------------------------------------


def retrieve(query_embedding, top_k):

    similarities = np.dot(
        passage_embeddings,
        query_embedding
    )

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        score = float(similarities[index])

        results.append({
            "index": int(passage_ids[index]),
            "text": passages[index],
            "score": max(0.0, min(1.0, score))
        })

    return results


# --------------------------------------------------
# SENTENCE EXTRACTION
# --------------------------------------------------


def extract_supporting_sentences(question, query_embedding, results):
    """Select evidence using semantic, passage, and keyword relevance."""
    stop_words = {
        "what", "which", "who", "when", "where", "does", "do",
        "is", "are", "the", "a", "an", "and", "or", "of",
        "to", "for", "in", "on", "use", "uses", "used",
        "with", "how", "much", "many"
    }
    question_terms = {
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9]+", question)
        if word.lower() not in stop_words
    }
    sentences = []
    for passage in results:
        parts = re.split(r"(?<=[.!?])\s+", passage["text"])
        for sentence in parts:
            sentence = sentence.strip()
            if len(sentence) < 10 or is_suspicious_sentence(sentence):
                continue

            sentence_terms = {
                word.lower()
                for word in re.findall(r"[a-zA-Z0-9]+", sentence)
            }
            overlap = sum(
                1
                for term in question_terms
                if term in sentence_terms
                or term + "s" in sentence_terms
                or term + "es" in sentence_terms
            )
            lexical_score = overlap / max(len(question_terms), 1)

            sentences.append({
                "text": sentence,
                "passage_index": passage["index"],
                "passage_score": passage["score"],
                "lexical_score": lexical_score
            })

    if not sentences:
        return []

    sentence_texts = [
        item["text"] for item in sentences
    ]

    sentence_embeddings = encode_texts(sentence_texts)

    semantic_scores = np.dot(
        sentence_embeddings,
        query_embedding
    )

    for item, score in zip(sentences, semantic_scores):
        item["sentence_score"] = (
            0.25 * float(score)
            + 0.65 * item["passage_score"]
            + 0.1 * item["lexical_score"]
        )

    sentences.sort(
        key=lambda item: item["sentence_score"],
        reverse=True
    )

    selected = []

    for item in sentences:
        if item["sentence_score"] < SENTENCE_THRESHOLD:
            continue

        if any(
            existing["text"] == item["text"]
            for existing in selected
        ):
            continue

        selected.append(item)

        if len(selected) == 3:
            break

    return selected


# --------------------------------------------------
# ANSWER GENERATION
# --------------------------------------------------

def generate_answer(question, query_embedding, results):

    if not results:
        return None, [], 0.0

    if results[0]["score"] < ABSTAIN_THRESHOLD:
        return None, [], 0.0

    evidence = extract_supporting_sentences(
        question,
        query_embedding,
        results
    )

    if not evidence:
        return None, [], 0.0

    answer = " ".join(
        item["text"] for item in evidence
    )

    used_indices = []

    for item in evidence:
        if item["passage_index"] not in used_indices:
            used_indices.append(item["passage_index"])

    confidence = min(
        0.99,
        max(
            0.61,
            0.65 * results[0]["score"]
            + 0.35 * evidence[0]["sentence_score"]
        )
    )

    return answer, used_indices, confidence


# --------------------------------------------------
# QUERY ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "RAG Colosseum API is running",
        "endpoint": "POST /query"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "passages": len(passages),
        "model_loaded": model is not None
    }


@app.post("/query")
def query(request: QueryRequest):

    if model is None or passage_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready"
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    query_embedding = encode_texts([question])[0]

    results = retrieve(
        query_embedding,
        request.top_k
    )

    answer, used_sources, confidence = generate_answer(
        question,
        query_embedding,
        results
    )

    if answer is None:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.1
        }

    # Only return sources that contributed to the answer.
    score_lookup = {
        item["index"]: item["score"]
        for item in results
    }

    used_sources.sort(
        key=lambda index: score_lookup[index],
        reverse=True
    )

    scores = [
        round(score_lookup[index], 4)
        for index in used_sources
    ]

    return {
        "answer": answer,
        "sources": used_sources,
        "scores": scores,
        "confidence": round(confidence, 4)
    }
