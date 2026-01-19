from pathlib import Path
import time
import os

import streamlit as st
import inngest
from dotenv import load_dotenv
import requests

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="RAG Ingest PDF",
    page_icon="📄",
    layout="centered",
)

# ------------------------------------------------------------------
# Inngest Client (cached)
# ------------------------------------------------------------------

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_app",
        is_production=False,
    )

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    client.send_sync(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    ids = client.send_sync(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return ids[0]


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(
    event_id: str,
    timeout_s: float = 120.0,
    poll_interval_s: float = 0.5,
) -> dict:
    start = time.time()
    last_status = None

    while True:
        runs = fetch_runs(event_id)

        if runs:
            run = runs[-1]  # always take the latest run
            status = run.get("status")
            last_status = status or last_status

            if status == "COMPLETED":
                return run.get("output") or {}

            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Inngest run {status}")

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output (last status: {last_status})"
            )

        time.sleep(poll_interval_s)

# ------------------------------------------------------------------
# UI: PDF Ingestion
# ------------------------------------------------------------------

st.title("Upload a PDF to Ingest")

uploaded = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        send_rag_ingest_event(path)
        time.sleep(0.3)

    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

# ------------------------------------------------------------------
# UI: RAG Query
# ------------------------------------------------------------------

st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input(
        "How many chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            try:
                if "rag_event_id" not in st.session_state:
                    st.session_state.rag_event_id = send_rag_query_event(
                        question.strip(),
                        int(top_k),
                    )

                output = wait_for_run_output(
                    st.session_state.rag_event_id,
                    timeout_s=120.0,
                )

                answer = output.get("answer", "")
                sources = output.get("sources", [])

            except Exception as e:
                st.error(f"Error while processing request: {e}")
                st.session_state.pop("rag_event_id", None)
                st.stop()

        st.subheader("Answer")
        st.write(answer or "(No answer returned)")

        if sources:
            st.caption("Sources")
            for src in sources:
                st.write(f"- {src}")

        # Clear state so next question triggers a fresh event
        st.session_state.pop("rag_event_id", None)
