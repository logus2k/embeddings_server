"""embeddings_server — one GPU (torch) service that serves bge models for the
bulário stack, replacing llama.cpp's embedding+rerank duty:

  POST /v1/rerank  {model?, query, documents[], top_n?}
        -> {results:[{index, relevance_score}]}  (sorted desc; Jina/Cohere contract)
        bge-reranker-v2-m3, fp16, DYNAMIC MICRO-BATCHING (coalesces concurrent
        requests into one GPU forward).

  POST /embed      {texts[], dense?=true, sparse?=false}
        -> {vectors:[[...]], sparse:[{indices,weights}]}
        bge-m3 dense (CLS-pooled, L2-normalized) + learned sparse (sparse_linear
        head). Only active when EMBED_ENABLED=true (VRAM: ~1.1 GB fp16). The
        thing llama.cpp CANNOT do is the sparse vectors.

  GET  /health

Env:
  RERANK_MODEL_PATH        (default on-disk bge-reranker-v2-m3)
  RERANK_MAX_LENGTH        per-pair token cap (default 512; model max 8192)
  RERANK_MAX_BATCH         max pairs per rerank forward (default 64, VRAM bound)
  RERANK_BATCH_WINDOW_MS   coalescing window (default 5 ms)
  EMBED_ENABLED            load bge-m3 for /embed (default false)
  EMBED_MODEL_PATH         (default on-disk bge-m3)
  EMBED_MAX_LENGTH         (default 512; model max 8192)
"""
import os, time, asyncio
from contextlib import asynccontextmanager
from typing import List, Optional
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ---- reranker -------------------------------------------------------------
RR_PATH = os.getenv("RERANK_MODEL_PATH",
                    "/models/bge-reranker-v2-m3")
RR_MAXLEN = int(os.getenv("RERANK_MAX_LENGTH", "512"))
RR_MAXBATCH = int(os.getenv("RERANK_MAX_BATCH", "64"))
RR_WINDOW = float(os.getenv("RERANK_BATCH_WINDOW_MS", "5")) / 1000.0

# ---- embeddings (optional) ------------------------------------------------
EMBED_ENABLED = os.getenv("EMBED_ENABLED", "false").lower() in ("1", "true", "yes")
EM_PATH = os.getenv("EMBED_MODEL_PATH", "/models/bge-m3")
EM_MAXLEN = int(os.getenv("EMBED_MAX_LENGTH", "512"))
# Embeddings can run on a SEPARATE device from the reranker. Default CPU so the
# sparse model adds 0 VRAM (query-time sparse = one short text; the fleet keeps
# dense on llama-vision). Set EMBED_DEVICE=cuda for bulk/ingestion throughput.
EM_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
_EM_DTYPE = torch.float16 if EM_DEVICE == "cuda" else torch.float32

print(f"[emb-srv] device={DEVICE} rerank={RR_PATH} embed_enabled={EMBED_ENABLED}", flush=True)
_t = time.time()
_rr_tok = AutoTokenizer.from_pretrained(RR_PATH)
_rr_model = AutoModelForSequenceClassification.from_pretrained(RR_PATH, dtype=_DTYPE).to(DEVICE).eval()
print(f"[emb-srv] reranker ready in {time.time()-_t:.1f}s "
      f"(max_len={RR_MAXLEN} max_batch={RR_MAXBATCH} window={RR_WINDOW*1000:.0f}ms)", flush=True)

_em_tok = _em_model = _sparse_linear = None
if EMBED_ENABLED:
    _t = time.time()
    _em_tok = AutoTokenizer.from_pretrained(EM_PATH)
    _em_model = AutoModel.from_pretrained(EM_PATH, dtype=_EM_DTYPE).to(EM_DEVICE).eval()
    # learned-sparse head: Linear(hidden, 1); weights in sparse_linear.pt
    sd = torch.load(os.path.join(EM_PATH, "sparse_linear.pt"), map_location=EM_DEVICE)
    _sparse_linear = torch.nn.Linear(_em_model.config.hidden_size, 1)
    _sparse_linear.load_state_dict(sd)
    _sparse_linear = _sparse_linear.to(EM_DEVICE).to(_EM_DTYPE).eval()
    print(f"[emb-srv] bge-m3 (dense+sparse) ready in {time.time()-_t:.1f}s on {EM_DEVICE}", flush=True)


# ==== reranker: dynamic micro-batching =====================================
@torch.inference_mode()
def _rr_score(pairs: List[List[str]]) -> List[float]:
    inp = _rr_tok(pairs, padding=True, truncation=True, max_length=RR_MAXLEN,
                  return_tensors="pt").to(DEVICE)
    return _rr_model(**inp).logits.view(-1).float().cpu().tolist()


class _Job:
    __slots__ = ("pairs", "future")
    def __init__(self, query, docs, loop):
        self.pairs = [[query, d] for d in docs]
        self.future = loop.create_future()


_queue: "asyncio.Queue[_Job]" = None


async def _batcher():
    loop = asyncio.get_running_loop()
    while True:
        job = await _queue.get()
        jobs = [job]; n = len(job.pairs); deadline = loop.time() + RR_WINDOW
        while n < RR_MAXBATCH:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                nxt = await asyncio.wait_for(_queue.get(), timeout)
            except asyncio.TimeoutError:
                break
            jobs.append(nxt); n += len(nxt.pairs)
        flat: List[List[str]] = []; spans = []
        for j in jobs:
            spans.append((j, len(flat), len(flat) + len(j.pairs))); flat.extend(j.pairs)
        scores: List[float] = []
        for i in range(0, len(flat), RR_MAXBATCH):
            scores.extend(await asyncio.to_thread(_rr_score, flat[i:i + RR_MAXBATCH]))
        for j, a, b in spans:
            if not j.future.done():
                j.future.set_result(scores[a:b])


# ==== embeddings ===========================================================
@torch.inference_mode()
def _embed(texts: List[str], dense: bool, sparse: bool):
    inp = _em_tok(texts, padding=True, truncation=True, max_length=EM_MAXLEN,
                  return_tensors="pt").to(EM_DEVICE)
    out = _em_model(**inp)
    hidden = out.last_hidden_state                       # [B, T, H]
    mask = inp["attention_mask"]                         # [B, T]
    vectors = sparse_out = None
    if dense:
        cls = hidden[:, 0]                               # bge-m3 dense = CLS token
        vectors = torch.nn.functional.normalize(cls.float(), p=2, dim=-1).cpu().tolist()
    if sparse:
        w = torch.relu(_sparse_linear(hidden).squeeze(-1))   # [B, T] token weights
        w = (w * mask).float()
        ids = inp["input_ids"]
        special = set(_em_tok.all_special_ids)
        sparse_out = []
        for b in range(ids.shape[0]):
            agg = {}
            for tid, wt in zip(ids[b].tolist(), w[b].tolist()):
                if wt <= 0 or tid in special:
                    continue
                if wt > agg.get(tid, 0.0):
                    agg[tid] = wt                        # max-pool per token id
            items = sorted(agg.items())
            sparse_out.append({"indices": [k for k, _ in items],
                               "weights": [round(v, 5) for _, v in items]})
    return vectors, sparse_out


# ==== app ==================================================================
@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _queue
    _queue = asyncio.Queue()
    task = asyncio.create_task(_batcher())
    yield
    task.cancel()


app = FastAPI(lifespan=_lifespan)


class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str
    documents: List[str]
    top_n: Optional[int] = None


class EmbedRequest(BaseModel):
    texts: List[str]
    dense: bool = True
    sparse: bool = False


@app.post("/v1/rerank")
async def rerank(req: RerankRequest):
    if not req.documents:
        return {"model": req.model or "bge-reranker-v2-m3", "object": "list",
                "usage": {"prompt_tokens": 0, "total_tokens": 0}, "results": []}
    job = _Job(req.query, req.documents, asyncio.get_running_loop())
    await _queue.put(job)
    scores = await job.future
    ranked = sorted(enumerate(scores), key=lambda p: -p[1])
    if req.top_n:
        ranked = ranked[:req.top_n]
    return {"model": req.model or "bge-reranker-v2-m3", "object": "list",
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
            "results": [{"index": i, "relevance_score": s} for i, s in ranked]}


@app.post("/embed")
async def embed(req: EmbedRequest):
    if not EMBED_ENABLED:
        raise HTTPException(503, "embeddings disabled (set EMBED_ENABLED=true)")
    if not req.texts:
        return {"vectors": [], "sparse": []}
    vectors, sparse = await asyncio.to_thread(_embed, req.texts, req.dense, req.sparse)
    resp = {}
    if req.dense:
        resp["vectors"] = vectors
    if req.sparse:
        resp["sparse"] = sparse
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "rerank_model": RR_PATH,
            "rerank_max_batch": RR_MAXBATCH, "embed_enabled": EMBED_ENABLED,
            "embed_model": EM_PATH if EMBED_ENABLED else None,
            "embed_device": EM_DEVICE if EMBED_ENABLED else None}
