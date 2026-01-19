from typing import List
import random
import logging

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Match this to your Qdrant collection dimension
EMBED_DIM = 3072

# Chunking config
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str) -> List[str]:
    """
    Load a PDF from disk and split it into text chunks.
    """
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks: List[str] = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    DEBUG VERSION: Fake embeddings, NO OpenAI calls.
    This avoids quota / rate-limit issues completely.
    """
    if not texts:
        return []

    vecs: List[List[float]] = []
    for t in texts:
        seed = abs(hash(t)) % (10**6)
        random.seed(seed)
        vecs.append([random.random() for _ in range(EMBED_DIM)])

    logger.warning("Using FAKE embeddings for %d texts", len(texts))
    return vecs
