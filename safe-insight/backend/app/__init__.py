"""
Safe Insight backend package.

Module map (read in this order to understand the pipeline):

* :mod:`app.config`         - all tunables and paths; imported first everywhere.
* :mod:`app.network_guard`  - socket kill-switch; installed before ML imports.
* :mod:`app.ingestion`      - PDF/DOCX/TXT -> text segments -> chunks.
* :mod:`app.chunking`       - the chunking algorithm itself.
* :mod:`app.embeddings`     - local sentence-transformers encoder.
* :mod:`app.vector_store`   - persistent FAISS index + metadata sidecar.
* :mod:`app.llm`            - local GGUF generation and grounded prompting.
* :mod:`app.citations`      - retrieval results -> verifiable citation trail.
* :mod:`app.audit_log`      - local SQLite compliance log.
* :mod:`app.schemas`        - HTTP request/response models.
* :mod:`app.main`           - FastAPI app and routes.
"""

__version__ = "0.1.0"
