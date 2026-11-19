
import json
import math
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
QUERY_EXPANSIONS = {
    "sgd": "stochastic gradient descent"
}
CONCEPT_ALIASES = {
    "sgd": "stochastic gradient descent",
    "stochastic gradient descent": "stochastic gradient descent",
    "adamw": "adamw",
    "raft": "raft",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mysql": "mysql"
}
STOP_WORDS = {
    "what", "which", "who", "when", "where", "does", "do",
    "is", "are", "the", "a", "an", "and", "or", "of",
    "to", "for", "in", "on", "use", "uses", "used",
    "with", "how", "much", "many", "compare", "between",
    "versus", "vs"
}

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


def expand_query(question):
    expanded_question = question

    for abbreviation, expansion in QUERY_EXPANSIONS.items():
        expanded_question = re.sub(
            rf"\b{re.escape(abbreviation)}\b",
            expansion,
            expanded_question,
            flags=re.IGNORECASE
        )

    return expanded_question


def normalize_term(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def get_terms(text):
    terms = set()
    for word in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        if word in STOP_WORDS:
            continue
        term = normalize_term(word)
        if term not in STOP_WORDS:
            terms.add(term)
    return terms


def get_term_weights(question):
    question_terms = get_terms(expand_query(question))
    if not question_terms:
        return {}

    document_frequencies = {}
    for passage in passages:
        for term in get_terms(passage):
            document_frequencies[term] = (
                document_frequencies.get(term, 0) + 1
            )

    document_count = max(len(passages), 1)
    return {
        term: 1.0 + math.log(
            (document_count + 1)
            / (document_frequencies.get(term, 0) + 1)
        )
        for term in question_terms
    }


def lexical_relevance(text, term_weights):
    if not term_weights:
        return 0.0

    text_terms = get_terms(text)
    matched_weight = sum(
        weight
        for term, weight in term_weights.items()
        if term in text_terms
    )
    return matched_weight / sum(term_weights.values())


def get_question_concepts(question):
    concepts = []
    for alias, concept in CONCEPT_ALIASES.items():
        if (
            concept not in concepts
            and re.search(
                rf"\b{re.escape(alias)}\b",
                question,
                flags=re.IGNORECASE
            )
        ):
            concepts.append(concept)
    return concepts


def get_fact_groups(question):
    terms = get_terms(expand_query(question))
    fact_groups = []

    if "batch" in terms or "size" in terms:
        fact_groups.append({"batch", "size", "sample", "example"})
    if "learning" in terms or "rate" in terms:
        fact_groups.append({"learning", "rate"})
    if "port" in terms:
        fact_groups.append({"port"})

    return fact_groups


# --------------------------------------------------
# SEMANTIC RETRIEVAL
# --------------------------------------------------


def retrieve(question, query_embedding, top_k):

    similarities = np.dot(
        passage_embeddings,
        query_embedding
    )

    term_weights = get_term_weights(question)
    passage_scores = []
    for passage, similarity in zip(passages, similarities):
        semantic_score = max(0.0, min(1.0, float(similarity)))
        lexical_score = lexical_relevance(passage, term_weights)
        passage_scores.append(
            0.65 * semantic_score + 0.35 * lexical_score
        )

    ranked_indices = np.argsort(passage_scores)[::-1].tolist()
    concepts = get_question_concepts(question)
    concept_indices = []
    for concept in concepts:
        matching_indices = [
            index
            for index, passage in enumerate(passages)
            if concept in passage.lower()
        ]
        if matching_indices:
            concept_indices.append(
                max(
                    matching_indices,
                    key=lambda index: passage_scores[index]
                )
            )

    if concept_indices:
        selected_indices = concept_indices[:top_k]
    else:
        selected_indices = ranked_indices[:top_k]

    if len(concept_indices) > 1 and len(selected_indices) < top_k:
        for index in ranked_indices:
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) == top_k:
                break

    selected_indices.sort(
        key=lambda index: passage_scores[index],
        reverse=True
    )

    results = []

    for index in selected_indices:
        results.append({
            "index": int(passage_ids[index]),
            "text": passages[index],
            "score": passage_scores[index]
        })

    return results


# --------------------------------------------------
# SENTENCE EXTRACTION
# --------------------------------------------------


def extract_supporting_sentences(question, query_embedding, results):
    """Select evidence using semantic, passage, and keyword relevance."""
    term_weights = get_term_weights(question)
    sentences = []
    for passage in results:
        parts = re.split(r"(?<=[.!?])\s+", passage["text"])
        for sentence in parts:
            sentence = sentence.strip()
            if len(sentence) < 10 or is_suspicious_sentence(sentence):
                continue

            sentences.append({
                "text": sentence,
                "passage_index": passage["index"],
                "passage_score": passage["score"],
                "lexical_score": lexical_relevance(sentence, term_weights)
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
            0.45 * max(0.0, min(1.0, float(score)))
            + 0.25 * item["passage_score"]
            + 0.3 * item["lexical_score"]
        )

    sentences.sort(
        key=lambda item: item["sentence_score"],
        reverse=True
    )

    concepts = get_question_concepts(question)
    if len(concepts) > 1:
        selected = []
        fact_groups = get_fact_groups(question)
        for concept in concepts:
            concept_passages = [
                passage
                for passage in results
                if concept in passage["text"].lower()
            ]
            if not concept_passages:
                continue

            target_index = max(
                concept_passages,
                key=lambda passage: passage["score"]
            )["index"]
            passage_sentences = [
                item for item in sentences
                if item["passage_index"] == target_index
            ]
            concept_sentences = [
                item for item in passage_sentences
                if item["sentence_score"] >= SENTENCE_THRESHOLD
            ]

            if fact_groups:
                for fact_group in fact_groups:
                    matching_sentences = [
                        item for item in passage_sentences
                        if get_terms(item["text"]) & fact_group
                    ]
                    numeric_sentences = [
                        item for item in matching_sentences
                        if re.search(
                            r"(?<![A-Za-z0-9])\d+(?:\.\d+)?",
                            item["text"]
                        )
                    ]
                    if numeric_sentences:
                        matching_sentences = numeric_sentences
                    if matching_sentences:
                        selected.append(max(
                            matching_sentences,
                            key=lambda item: item["sentence_score"]
                        ))
            else:
                selected.extend(concept_sentences[:3])

        if selected:
            return list({
                item["text"]: item
                for item in selected
            }.values())

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

    if is_suspicious_sentence(question):
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.1
        }

    query_embedding = encode_texts([expand_query(question)])[0]

    results = retrieve(
        question,
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
