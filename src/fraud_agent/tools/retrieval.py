"""RAG over the fraud playbook + long-term memory of past cases (pgvector via langchain-postgres).

    uv run python -m fraud_agent.tools.retrieval --ingest     # index data/playbook/*.md
"""
import argparse
from functools import lru_cache
from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from langchain_text_splitters import MarkdownHeaderTextSplitter

from fraud_agent.config import get_settings

PLAYBOOK_DIR = Path("data/playbook")


class FastEmbedEmbeddings(Embeddings):
    """Local ONNX embeddings (no API key, no torch). Implements LangChain's Embeddings interface."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache
def embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings()


@lru_cache
def store(collection: str) -> PGVector:
    return PGVector(embeddings=embeddings(), collection_name=collection,
                    connection=get_settings().sqlalchemy_dsn, use_jsonb=True)


def ingest_playbook() -> int:
    splitter = MarkdownHeaderTextSplitter([("#", "doc"), ("##", "section")], strip_headers=False)
    docs: list[Document] = []
    for path in sorted(PLAYBOOK_DIR.glob("*.md")):
        for chunk in splitter.split_text(path.read_text(encoding="utf-8")):
            chunk.metadata["source"] = path.name
            docs.append(chunk)
    vs = PGVector(embeddings=embeddings(), collection_name="playbook",
                  connection=get_settings().sqlalchemy_dsn, use_jsonb=True,
                  pre_delete_collection=True)
    vs.add_documents(docs)
    return len(docs)


def search_playbook(query: str, k: int = 3) -> list[dict]:
    hits = store("playbook").similarity_search_with_score(query, k=k)
    return [{"ref": f"{d.metadata.get('source')} > {d.metadata.get('section', d.metadata.get('doc'))}",
             "distance": round(score, 3), "text": d.page_content} for d, score in hits]


def remember_case(case_id: str, summary: str, metadata: dict) -> None:
    store("case_memory").add_documents([Document(page_content=summary,
                                                 metadata={"case_id": case_id, **metadata})],
                                       ids=[case_id])


def search_cases(query: str, k: int = 3) -> list[dict]:
    hits = store("case_memory").similarity_search_with_score(query, k=k)
    return [{"case_id": d.metadata.get("case_id"), "verdict": d.metadata.get("verdict"),
             "pattern": d.metadata.get("fraud_pattern"), "reviewer_decision": d.metadata.get("decision"),
             "distance": round(score, 3), "summary": d.page_content} for d, score in hits]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--query")
    a = ap.parse_args()
    if a.ingest:
        print(f"Indexed {ingest_playbook()} playbook chunks")
    if a.query:
        for h in search_playbook(a.query):
            print(h["distance"], h["ref"])
