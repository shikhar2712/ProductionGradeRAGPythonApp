import logging
import uuid
import datetime

from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

logger = logging.getLogger("uvicorn")

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

# ------------ RAG: Ingest PDF ------------
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2,
        period=datetime.timedelta(minutes=1),
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context, step: inngest.Step):
    """
    Ingest a PDF: load, chunk, embed (fake), and upsert into Qdrant.
    """

    def _load() -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        vecs = embed_texts(chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {"source": source_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]

        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src: RAGChunkAndSrc = await step.run(
        "load-and-chunk",
        _load,
        output_type=RAGChunkAndSrc,
    )

    ingested: RAGUpsertResult = await step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    return ingested.model_dump()


# ------------ RAG: Query PDF ------------
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context, step: inngest.Step):
    """
    DEBUG VERSION: Only Qdrant search + fake embeddings.
    NO LLM / OpenAI here, so it should always finish quickly.
    """

    def _search() -> RAGSearchResult:
        question = ctx.event.data["question"]
        top_k = int(ctx.event.data.get("top_k", 5))

        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)

        return RAGSearchResult(
            contexts=found.get("contexts", []),
            sources=found.get("sources", []),
        )

    found: RAGSearchResult = await step.run(
        "embed-and-search",
        _search,
        output_type=RAGSearchResult,
    )

    # Minimal debug-style answer
    question = ctx.event.data["question"]
    answer = f"(DEBUG) You asked: {question!r}. Retrieved {len(found.contexts)} chunks."

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


# ------------ FastAPI wiring ------------
app = FastAPI()

inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf_ai],
)
