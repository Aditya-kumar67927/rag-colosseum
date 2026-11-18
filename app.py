
import json
import re
import threading
from pathlib import Path

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer


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
MODEL_LOCK = threading.Lock()


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

@app.on_event("startup")
def initialize():

    global model
    global passages
    global passage_embeddings
    global passage_ids

    print("Loading knowledge base...")

    passages, passage_ids = load_knowledge_base()

    print("Loaded passages:", len(passages))

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Generating passage embeddings...")

    with MODEL_LOCK:
        passage_embeddings = model.encode(
            passages,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

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

def retrieve(question, top_k):

    with MODEL_LOCK:
        query_embedding = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

    similarities = np.dot(
        passage_embeddings,
        query_embedding
    )

    # Return original 0-based passage indices.
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

def extract_supporting_sentences(question, results):

    sentences = []

    for passage in results:

        parts = re.split(
            r"(?<=[.!?])\s+",
            passage["text"]
        )

        for sentence in parts:

            sentence = sentence.strip()

            if len(sentence) < 10:
                continue

            # Do not execute or repeat suspicious instructions.
            if is_suspicious_sentence(sentence):
                continue

            sentences.append({
                "text": sentence,
                "passage_index": passage["index"],
                "passage_score": passage["score"]
            })

    if not sentences:
        return []

    sentence_texts = [
        item["text"] for item in sentences
    ]

    with MODEL_LOCK:
        sentence_embeddings = model.encode(
            sentence_texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        query_embedding = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

    sentence_scores = np.dot(
        sentence_embeddings,
        query_embedding
    )

    for item, score in zip(sentences, sentence_scores):
        item["sentence_score"] = float(score)

    sentences.sort(
        key=lambda item: item["sentence_score"],
        reverse=True
    )

    selected = []

    # Keep the strongest distinct supporting sentences.
    for item in sentences:

        if item["sentence_score"] < SENTENCE_THRESHOLD:
            continue

        if item["text"] in [
            existing["text"] for existing in selected
        ]:
            continue

        selected.append(item)

        if len(selected) == 3:
            break

    return selected


# --------------------------------------------------
# ANSWER GENERATION
# --------------------------------------------------

def generate_answer(question, results):

    if not results:
        return None, [], 0.0

    if results[0]["score"] < ABSTAIN_THRESHOLD:
        return None, [], 0.0

    evidence = extract_supporting_sentences(
        question,
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

    results = retrieve(
        question,
        request.top_k
    )

    answer, used_sources, confidence = generate_answer(
        question,
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
