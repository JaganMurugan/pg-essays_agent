#!/usr/bin/env python3
"""
Paul Graham essays -> chunks -> embeddings -> Pinecone (resumable).

INSTALL
    uv add datasets sentence-transformers pinecone tiktoken
    uv add python-dotenv     # optional: auto-loads a .env file
    uv add openai            # only if you use EMBED_PROVIDER=openai

CONFIGURE (environment variables, or a .env file in the directory you run from)
    PINECONE_API_KEY    required
    OPENAI_API_KEY      required only when EMBED_PROVIDER=openai
    EMBED_PROVIDER      "local" (default: free BGE-M3 on your machine) or "openai"
    PG_MD_DIR           ingest a folder of .md files instead of the Hugging Face dataset
    PG_NAMESPACE        Pinecone namespace            (default: essays)
    PINECONE_INDEX      index name                    (default: pg-essays-bge-m3 / pg-essays-openai)
    PINECONE_CLOUD / PINECONE_REGION                  (default: aws / us-east-1)
    PG_CHUNK_TOKENS     max tokens per chunk          (default: 400)
    PG_OVERLAP_TOKENS   overlap between chunks        (default: 60)
    PG_BATCH            chunks per embed+upsert batch (default: 64)
    PG_CHECKPOINT       checkpoint file path          (default: pg_ingest_checkpoint_<index>.json)
    LOG_LEVEL           INFO (default) / DEBUG / WARNING

COMMANDS
    python pg_ingest.py --dry-run            # load + chunk only, print stats, touch nothing
    python pg_ingest.py                      # ingest; safe to re-run, resumes after a crash
    python pg_ingest.py --fresh              # ignore the checkpoint and redo everything
    python pg_ingest.py "how to find startup ideas" [--top-k 5]    # query the index

RESUMING
    Progress is saved after every batch. If the run dies (network, Ctrl-C, out of memory),
    run the same command again. The checkpoint is discarded automatically when the data,
    chunking settings, namespace or embedding model change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pathlib
import random
import re
import signal
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Iterator

try:  # optional convenience; ignored if python-dotenv isn't installed
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))  # a .env in the directory you run from (or above it)
except ImportError:
    pass

import tiktoken
from pinecone import Pinecone, ServerlessSpec

log = logging.getLogger("pg_ingest")

# ============================== configuration ==============================


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {raw!r}")
    if value <= 0:
        raise SystemExit(f"{name} must be positive, got {value}")
    return value


HF_DATASET = "sgoel9/paul_graham_essays"

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").strip().lower()
if EMBED_PROVIDER == "openai":
    EMBED_MODEL, EMBED_DIM, _DEFAULT_INDEX = "text-embedding-3-small", 1536, "pg-essays-openai"
elif EMBED_PROVIDER == "local":
    EMBED_MODEL, EMBED_DIM, _DEFAULT_INDEX = "BAAI/bge-m3", 1024, "pg-essays-bge-m3"
else:
    raise SystemExit(f"EMBED_PROVIDER must be 'local' or 'openai', got {EMBED_PROVIDER!r}")

INDEX_NAME = os.getenv("PINECONE_INDEX", _DEFAULT_INDEX)
NAMESPACE = os.getenv("PG_NAMESPACE", "essays")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

CHUNK_TOKENS = _env_int("PG_CHUNK_TOKENS", 400)
OVERLAP_TOKENS = _env_int("PG_OVERLAP_TOKENS", 60)
BATCH = _env_int("PG_BATCH", 64)  # kept modest so a 1536-dim upsert stays well under Pinecone's 2 MB request cap
if OVERLAP_TOKENS >= CHUNK_TOKENS:
    raise SystemExit("PG_OVERLAP_TOKENS must be smaller than PG_CHUNK_TOKENS")

CHECKPOINT_PATH = pathlib.Path(
    os.getenv("PG_CHECKPOINT", f"pg_ingest_checkpoint_{INDEX_NAME}.json")
)

# ================================ data model ================================


@dataclass(frozen=True)
class Essay:
    title: str
    date: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    embed_text: str
    metadata: dict


# ============================ logging / signals =============================


def setup_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _on_sigterm(signum, frame):  # `docker stop` / `kill` -> same clean path as Ctrl-C
    raise KeyboardInterrupt


# ================================ tokenizer =================================

_enc = None


def count_tokens(text: str) -> int:
    """Approximate token count (OpenAI cl100k). Close enough for sizing chunks for any model."""
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return len(_enc.encode(text))


# ============================== 1. load essays ==============================

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_MONTH_YEAR = re.compile(rf"^(?:{_MONTHS})\s+\d{{4}}$")
_FRONTMATTER = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*\n", re.DOTALL)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_hf() -> Iterator[Essay]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET)
    split = ds["train"] if "train" in ds else next(iter(ds.values()))
    missing = {"title", "text"} - set(split.column_names)
    if missing:
        raise RuntimeError(f"{HF_DATASET} is missing expected columns: {sorted(missing)}")
    for row in split:
        yield Essay(
            title=(row.get("title") or "").strip(),
            date=(row.get("date") or "").strip(),
            text=row.get("text") or "",
        )


def load_markdown_dir(path: str) -> Iterator[Essay]:
    """Essays from a folder of .md files. Title: first '# ' heading, else the file name."""
    root = pathlib.Path(path)
    if not root.is_dir():
        raise RuntimeError(f"PG_MD_DIR is not a directory: {path}")
    files = [
        p
        for p in sorted(root.rglob("*.md"))
        if p.name.lower() != "readme.md"
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    ]
    if not files:
        raise RuntimeError(f"no .md files found under {path}")
    for p in files:
        raw = _FRONTMATTER.sub("", p.read_text(encoding="utf-8", errors="replace"), count=1)
        lines = raw.lstrip().splitlines()
        title = p.stem.replace("-", " ").replace("_", " ").strip()
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip() or title
            lines = lines[1:]
        date = next((ln.strip() for ln in lines[:6] if _MONTH_YEAR.match(ln.strip())), "")
        yield Essay(title=title, date=date, text="\n".join(lines))


def load_essays() -> Iterator[Essay]:
    md_dir = os.getenv("PG_MD_DIR")
    return load_markdown_dir(md_dir) if md_dir else load_hf()


# ================================ 2. chunk ==================================


def _split_long(paragraph: str) -> list[str]:
    """Split a paragraph longer than CHUNK_TOKENS into sentence groups, preserving order."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    cur: list[str] = []
    for s in sentences:
        if count_tokens(s) > CHUNK_TOKENS:  # one monster sentence: hard-split by words
            if cur:
                pieces.append(" ".join(cur))
                cur = []
            words = s.split()
            step = max(1, int(CHUNK_TOKENS * 0.6))
            pieces.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
            continue
        if cur and count_tokens(" ".join(cur + [s])) > CHUNK_TOKENS:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_text(text: str) -> list[str]:
    """Paragraph-aware chunks of <= ~CHUNK_TOKENS, repeating a short tail as overlap."""
    units: list[tuple[str, int]] = []
    for para in (p.strip() for p in re.split(r"\n+", text)):
        if not para:
            continue
        t = count_tokens(para)
        if t <= CHUNK_TOKENS:
            units.append((para, t))
        else:
            units.extend((piece, count_tokens(piece)) for piece in _split_long(para))

    chunks: list[str] = []
    cur: list[tuple[str, int]] = []
    cur_tokens = 0
    for unit, t in units:
        if cur and cur_tokens + t > CHUNK_TOKENS:
            chunks.append("\n\n".join(u for u, _ in cur))
            carry: list[tuple[str, int]] = []
            carry_t = 0
            for prev in reversed(cur):
                if carry_t + prev[1] > OVERLAP_TOKENS:
                    break
                carry.insert(0, prev)
                carry_t += prev[1]
            cur, cur_tokens = carry, carry_t
        cur.append((unit, t))
        cur_tokens += t
    if cur:
        chunks.append("\n\n".join(u for u, _ in cur))
    return chunks


def slugify(title: str, fallback: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80] or fallback


def build_chunks(essays: Iterable[Essay]) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    n_essays = 0
    for n, essay in enumerate(essays):
        title = essay.title.strip() or f"essay-{n}"
        text = clean_text(essay.text)
        if not text:
            log.warning("skipping essay with no text: %r", title)
            continue
        slug = slugify(title, f"essay-{n}")
        if slug in seen:  # same title twice: keep ids unique (checkpoint relies on it)
            slug = f"{slug}-{n}"
        seen.add(slug)
        pieces = chunk_text(text)
        n_essays += 1
        for i, piece in enumerate(pieces):
            meta = {
                "essay_id": slug,
                "title": title,
                "chunk_index": i,
                "n_chunks": len(pieces),
                "text": piece,  # raw chunk, so a query result needs no second lookup
            }
            if essay.date:
                meta["date"] = essay.date
            chunks.append(
                # the title is embedded with the chunk so each chunk carries its essay's context
                Chunk(id=f"{slug}-{i}", embed_text=f"{title}\n\n{piece}", metadata=meta)
            )
    if not chunks:
        raise RuntimeError("no chunks were produced; is the data source empty?")
    log.info("built %d chunks from %d essays", len(chunks), n_essays)
    return chunks


def dry_run(chunks: list[Chunk]) -> None:
    sizes = [count_tokens(c.metadata["text"]) for c in chunks]
    essays = len({c.metadata["essay_id"] for c in chunks})
    log.info("essays: %d | chunks: %d | total tokens: %d", essays, len(chunks), sum(sizes))
    log.info(
        "chunk tokens: min %d | median %d | max %d (target <= %d)",
        min(sizes), int(statistics.median(sizes)), max(sizes), CHUNK_TOKENS,
    )
    log.info("provider=%s model=%s dim=%d index=%s namespace=%s",
             EMBED_PROVIDER, EMBED_MODEL, EMBED_DIM, INDEX_NAME, NAMESPACE)
    log.info("sample chunk -> %s", chunks[0].metadata["text"][:200].replace("\n", " "))


# ============================ retries / embedding ===========================


def _is_transient(exc: Exception) -> bool:
    """Retry rate limits, 5xx and network errors; don't retry auth/validation/our own bugs."""
    if isinstance(exc, (KeyError, TypeError, ValueError, AssertionError)):
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return status in (408, 409, 425, 429) or status >= 500
    return True  # connection resets, timeouts, ...


def with_retries(fn, what: str, retries: int = 5, base: float = 1.0):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries or not _is_transient(exc):
                raise
            wait = base * 2 ** (attempt - 1) * (0.5 + random.random())  # exponential + jitter
            log.warning("%s failed (%s); retry %d/%d in %.1fs", what, exc, attempt, retries - 1, wait)
            time.sleep(wait)


_embedder = None


def embed(texts: list[str]) -> list[list[float]]:
    global _embedder
    if EMBED_PROVIDER == "openai":
        if _embedder is None:
            from openai import OpenAI

            _embedder = OpenAI(max_retries=0, timeout=60)  # we handle retries ourselves
        resp = with_retries(
            lambda: _embedder.embeddings.create(model=EMBED_MODEL, input=texts), "embedding"
        )
        vectors = [d.embedding for d in resp.data]
    else:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading %s (first run downloads a couple of GB)", EMBED_MODEL)
            _embedder = SentenceTransformer(EMBED_MODEL)
            _embedder.max_seq_length = 1024  # chunks are ~400 tokens; leaves headroom
        vectors = _embedder.encode(texts, batch_size=16, normalize_embeddings=True).tolist()

    if len(vectors) != len(texts) or any(len(v) != EMBED_DIM for v in vectors):
        raise RuntimeError(f"embedder returned unexpected shapes (expected {EMBED_DIM}-dim vectors)")
    return vectors


# ================================ 3. Pinecone ===============================


def get_pinecone() -> Pinecone:
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def get_index(pc: Pinecone):
    if INDEX_NAME not in pc.list_indexes().names():
        log.info("creating index %s (dim=%d, cosine, %s/%s)", INDEX_NAME, EMBED_DIM, CLOUD, REGION)
        with_retries(
            lambda: pc.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            ),
            "create_index",
        )
    deadline = time.monotonic() + 300
    while True:
        desc = pc.describe_index(INDEX_NAME)
        if desc.status["ready"]:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"index {INDEX_NAME} was not ready after 5 minutes")
        time.sleep(2)
    if desc.dimension != EMBED_DIM:
        raise RuntimeError(
            f"index {INDEX_NAME} has dimension {desc.dimension} but {EMBED_MODEL} produces "
            f"{EMBED_DIM}. Use a different PINECONE_INDEX or delete the old index."
        )
    return pc.Index(INDEX_NAME)


def verify(index, expected: int) -> None:
    """Best-effort check that Pinecone holds the vectors. Stats are eventually consistent."""
    count = -1
    try:
        for _ in range(10):
            stats = index.describe_index_stats()
            ns = stats.namespaces.get(NAMESPACE)
            count = ns.vector_count if ns else 0
            if count >= expected:
                break
            time.sleep(3)
    except Exception as exc:  # never fail a finished run over a stats call
        log.warning("could not verify vector count: %s", exc)
        return
    if count == expected:
        log.info("verified: namespace %r holds %d vectors", NAMESPACE, count)
    elif count > expected:
        log.warning("namespace %r holds %d vectors but this run produced %d; stale vectors from an "
                    "earlier chunking may remain (delete the namespace to clean up)",
                    NAMESPACE, count, expected)
    else:
        log.warning("namespace %r reports %d/%d vectors; stats can lag, re-check in a minute",
                    NAMESPACE, count, expected)


# ============================== 4. checkpointing ============================


def fingerprint(chunks: list[Chunk]) -> str:
    """Hash of everything that determines the stored vectors."""
    h = hashlib.sha256(
        f"{EMBED_MODEL}|{EMBED_DIM}|{NAMESPACE}|{CHUNK_TOKENS}|{OVERLAP_TOKENS}".encode()
    )
    for c in chunks:
        h.update(c.id.encode())
        h.update(b"\0")
        h.update(c.embed_text.encode())
        h.update(b"\0")
    return h.hexdigest()


def load_checkpoint(path: pathlib.Path, fp: str) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("checkpoint unreadable; starting fresh")
        return set()
    if data.get("fingerprint") != fp:
        log.warning("checkpoint doesn't match current data/settings/model; starting fresh")
        return set()
    return set(data.get("done", []))


def save_checkpoint(path: pathlib.Path, fp: str, done: set[str]) -> None:
    tmp = path.with_suffix(".tmp")  # write then rename: a crash can't leave a half-written file
    tmp.write_text(json.dumps({"fingerprint": fp, "done": sorted(done)}))
    os.replace(tmp, path)


# ================================ 5. ingest =================================


def ingest(chunks: list[Chunk], fresh: bool = False) -> None:
    fp = fingerprint(chunks)
    if fresh and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    done = load_checkpoint(CHECKPOINT_PATH, fp)
    todo = [c for c in chunks if c.id not in done]
    log.info("%d chunks total | %d already done | %d to go", len(chunks), len(done), len(todo))
    if not todo:
        log.info("nothing to do (use --fresh to redo everything, e.g. after deleting the index)")
        return

    index = get_index(get_pinecone())
    started = time.monotonic()
    processed = 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        vectors = embed([c.embed_text for c in batch])
        payload = [
            {"id": c.id, "values": v, "metadata": c.metadata} for c, v in zip(batch, vectors)
        ]
        with_retries(lambda: index.upsert(vectors=payload, namespace=NAMESPACE), "upsert")

        done.update(c.id for c in batch)  # record progress only once the batch is in Pinecone
        save_checkpoint(CHECKPOINT_PATH, fp, done)
        processed += len(batch)
        rate = processed / max(time.monotonic() - started, 1e-9)
        log.info("progress %d/%d | %.1f chunks/s | ETA %ds",
                 len(done), len(chunks), rate, int((len(todo) - processed) / rate))

    log.info("ingest finished")
    verify(index, len(chunks))


# ================================ 6. query ==================================


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Top-k chunks for a query: [{'score', 'title', 'text', 'chunk_index', ...}, ...]."""
    index = get_pinecone().Index(INDEX_NAME)
    qvec = embed([query])[0]
    res = with_retries(
        lambda: index.query(vector=qvec, top_k=top_k, include_metadata=True, namespace=NAMESPACE),
        "query",
    )
    return [{"score": m.score, **(m.metadata or {})} for m in res.matches]


def print_results(results: list[dict]) -> None:
    if not results:
        print("no results (has the index been populated?)")
    for r in results:
        where = f"chunk {r.get('chunk_index', 0) + 1}/{r.get('n_chunks', '?')}"
        date = f" ({r['date']})" if r.get("date") else ""
        print(f"\n{r['score']:.3f}  {r.get('title', '?')}{date}  {where}")
        print(r.get("text", "")[:300].replace("\n", " ") + "...")


# ================================== main ====================================


def check_env(need_pinecone: bool) -> None:
    missing = []
    if need_pinecone and not os.getenv("PINECONE_API_KEY"):
        missing.append("PINECONE_API_KEY")
    if need_pinecone and EMBED_PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError(f"missing environment variable(s): {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    signal.signal(signal.SIGTERM, _on_sigterm)

    ap = argparse.ArgumentParser(description="Ingest Paul Graham's essays into Pinecone, or query them.")
    ap.add_argument("query", nargs="*", help="if given, search the index instead of ingesting")
    ap.add_argument("--top-k", type=int, default=5, help="results to return for a query")
    ap.add_argument("--fresh", action="store_true", help="ignore the checkpoint and redo everything")
    ap.add_argument("--dry-run", action="store_true", help="load and chunk only; print stats")
    args = ap.parse_args(argv)

    try:
        check_env(need_pinecone=not args.dry_run)
        if args.query:
            print_results(retrieve(" ".join(args.query), args.top_k))
        else:
            chunks = build_chunks(load_essays())
            if args.dry_run:
                dry_run(chunks)
            else:
                ingest(chunks, fresh=args.fresh)
        return 0
    except KeyboardInterrupt:
        log.warning("interrupted; progress up to the last finished batch is saved. Re-run to resume.")
        return 130
    except Exception:
        log.exception("failed; progress up to the last finished batch is saved. Re-run to resume.")
        return 1


if __name__ == "__main__":
    sys.exit(main())