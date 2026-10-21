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
  RERANK_DEVICE            cuda / xpu / cpu (default: the GPU PyTorch finds - cuda, then xpu - else cpu)
  RERANK_MODEL_PATH        (default on-disk bge-reranker-v2-m3)
  RERANK_MAX_LENGTH        per-pair token cap (default 512; model max 8192)
  RERANK_MAX_BATCH         max pairs per rerank forward (default 64, VRAM bound)
  RERANK_BATCH_WINDOW_MS   coalescing window (default 5 ms)
  EMBED_ENABLED            load bge-m3 for /embed (default false)
  EMBED_MODEL_PATH         (default on-disk bge-m3)
  EMBED_MAX_LENGTH         (default 512; model max 8192)
  EMBED_CONCURRENCY        /embed requests computed at once (default 4)

GPU memory given back when idle: PyTorch keeps the memory a batch needed reserved afterwards, at its peak (measured
2026-10-15: 4,106 MiB held after a day of ingestion, ~1.5 GB of it free blocks). After GPU_RELEASE_IDLE_SECONDS (120)
with no rerank or embed in flight, the free blocks go back to the card (empty_cache() of the device the models run on:
torch.cuda on NVIDIA, torch.xpu on an Intel Arc); the models stay.
GPU_RELEASE_IDLE (true) turns it off. GET /admin/gpu-memory reads it, PUT /admin/gpu-memory {release_when_idle?,
idle_seconds?} sets it (Cortex, from its settings). ADMIN_TOKEN, when set, is the bearer token these two routes need
(as ingestion-server's API_TOKEN: empty = open, a development posture) - the other routes stay open on the internal
networks, but ~30 containers share those, and any of them could change the release.

Threads: the reranker has its own worker thread and /embed its own pool. Both used to go through asyncio's
shared default pool, so a burst of /embed requests (CPU work) made /v1/rerank (GPU work) queue behind it:
64 embeds + 64 reranks at once -> reranks up to 14 s instead of 1.5 s (measured 2026-09-29). Embeds at once,
measured with 32 requests (wall) / the 64+64 burst (rerank worst): 1 -> 17.9 s / 1.8 s, 2 -> 16.4 / 2.7,
4 -> 9.6 / 2.2, 8 -> 9.0 / 2.6, unlimited -> 9.6 / 14.2. 4 keeps the embeds' speed and the rerank's
latency, and leaves CPU for others. A failing rerank batch fails its own requests; the batcher goes on
(before, one exception ended it and every later rerank waited forever).
"""
import os, time, asyncio, gc, threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional
import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def _gpu() -> str:
    """The GPU PyTorch has here: cuda (an NVIDIA build), xpu (Intel's build: Arc, also under WSL - Dockerfile.xpu),
    else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _dtype(device: str):
    return torch.float32 if device == "cpu" else torch.float16


DEVICE = os.getenv("RERANK_DEVICE") or _gpu()
_DTYPE = _dtype(DEVICE)

# ---- reranker -------------------------------------------------------------
RR_PATH = os.getenv("RERANK_MODEL_PATH",
                    "/models/bge-reranker-v2-m3")
RR_MAXLEN = int(os.getenv("RERANK_MAX_LENGTH", "512"))
RR_MAXBATCH = int(os.getenv("RERANK_MAX_BATCH", "64"))
RR_WINDOW = float(os.getenv("RERANK_BATCH_WINDOW_MS", "5")) / 1000.0
# padded tokens per forward pass, at most: a batch runs in groups of similar length that fit it, so a long text no longer
# pads a whole batch to its length (RERANK_MAX_BATCH x RERANK_MAX_LENGTH was the only bound; with the length raised to
# the model's 8192, 64 x 8192 could not fit beside the other models on a shared card). Default: the old worst case,
# 64 x 512 - batches of ordinary texts run exactly as before
RR_TOKEN_BUDGET = int(os.getenv("RERANK_TOKEN_BUDGET", str(64 * 512)))

# ---- embeddings (optional) ------------------------------------------------
EMBED_ENABLED = os.getenv("EMBED_ENABLED", "false").lower() in ("1", "true", "yes")
EM_PATH = os.getenv("EMBED_MODEL_PATH", "/models/bge-m3")
EM_MAXLEN = int(os.getenv("EMBED_MAX_LENGTH", "512"))
# Embeddings can run on a SEPARATE device from the reranker. Default CPU so the
# sparse model adds 0 VRAM (query-time sparse = one short text; the fleet keeps
# dense on llama-vision). Set EMBED_DEVICE=cuda (or xpu) for bulk/ingestion throughput.
EM_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
EMBED_CONCURRENCY = max(1, int(os.getenv("EMBED_CONCURRENCY", "4")))
EM_TOKEN_BUDGET = int(os.getenv("EMBED_TOKEN_BUDGET", str(32 * 512)))   # as RERANK_TOKEN_BUDGET, for /embed
_RR_POOL = ThreadPoolExecutor(1, thread_name_prefix="rerank")
_EM_POOL = ThreadPoolExecutor(EMBED_CONCURRENCY, thread_name_prefix="embed")
_EM_DTYPE = _dtype(EM_DEVICE)

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


# ==== GPU memory given back when idle =====================================
_GPU = {"release_when_idle": os.getenv("GPU_RELEASE_IDLE", "true").strip().lower() not in ("0", "false", "no", "off"),
        "idle_seconds": max(10, int(os.getenv("GPU_RELEASE_IDLE_SECONDS", "120")))}
_gpu_lock = threading.Lock()          # held while the work in flight is counted, and while memory is given back
_gpu_busy = [0]
_gpu_last = [time.monotonic()]
_gpu_released = [True]                # nothing used since the last release: nothing to give back
_gpu_freed_total = [0.0]


def _gpu_work(fn):
    """A function that runs on the GPU: counted while it runs, the idle clock restarted when it ends."""
    def run(*args, **kwargs):
        with _gpu_lock:
            _gpu_busy[0] += 1
        try:
            return fn(*args, **kwargs)
        finally:
            with _gpu_lock:
                _gpu_busy[0] -= 1
                _gpu_last[0] = time.monotonic()
                _gpu_released[0] = False
    return run


def _accelerators() -> list:
    """The GPU modules the models run on: torch.cuda (NVIDIA) and/or torch.xpu (Intel Arc: Dockerfile.xpu) - the
    reranker's DEVICE and, when /embed is on, EM_DEVICE. Both have the same memory calls (memory_reserved,
    memory_allocated, empty_cache). On the CPU, none."""
    used = {DEVICE} | ({EM_DEVICE} if EMBED_ENABLED else set())
    return [getattr(torch, d) for d in ("cuda", "xpu") if d in used and hasattr(torch, d) and getattr(torch, d).is_available()]


def _reserved() -> int:
    return sum(a.memory_reserved() for a in _accelerators())


def _gpu_release() -> float:
    """The free blocks handed back, when nothing is in flight; MiB given back (-1: something is running)."""
    accs = _accelerators()
    if not accs:
        _gpu_released[0] = True
        return 0.0
    with _gpu_lock:
        if _gpu_busy[0]:
            return -1.0
        before = _reserved()
        gc.collect()
        for a in accs:
            a.empty_cache()
        after = _reserved()
        _gpu_released[0] = True
    freed = (before - after) / 2**20
    _gpu_freed_total[0] += max(0.0, freed)
    print(f"[emb-srv] gpu memory: idle {time.monotonic() - _gpu_last[0]:.0f} s, {freed:.0f} MiB given back "
          f"({after / 2**20:.0f} MiB still reserved)", flush=True)
    return freed


async def _gpu_watch():
    while True:
        await asyncio.sleep(15)
        try:
            if (_GPU["release_when_idle"] and not _gpu_released[0] and not _gpu_busy[0]
                    and time.monotonic() - _gpu_last[0] >= _GPU["idle_seconds"]):
                await asyncio.get_running_loop().run_in_executor(None, _gpu_release)
        except Exception as e:  # noqa: BLE001 - the watcher never dies
            print(f"[emb-srv] gpu memory: release failed: {type(e).__name__}: {e}", flush=True)


def _gpu_state() -> dict:
    accs = _accelerators()
    return {**_GPU, "devices": [a.__name__.split(".")[-1] for a in accs],
            "reserved_mib": round(_reserved() / 2**20),
            "allocated_mib": round(sum(a.memory_allocated() for a in accs) / 2**20),
            "released_mib": round(_gpu_freed_total[0]), "idle_for_seconds": round(time.monotonic() - _gpu_last[0]),
            # the most allocated at once since the start (the batches' working memory): what a longer limit costs
            "peak_allocated_mib": round(sum(a.max_memory_allocated() for a in accs) / 2**20),
            "max_length": {"rerank": RR_MAXLEN, "embed": EM_MAXLEN},
            "token_budget": {"rerank": RR_TOKEN_BUDGET, "embed": EM_TOKEN_BUDGET}}


def _groups(lengths: List[int], budget: int) -> List[List[int]]:
    """The indices in groups of similar length whose padded size (count x longest) stays within `budget` - a text longer
    than the budget alone in its group."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out: List[List[int]] = []
    group: List[int] = []
    for i in order:
        if group and (len(group) + 1) * lengths[i] > budget:      # sorted: this one is the group's longest
            out.append(group)
            group = []
        group.append(i)
    if group:
        out.append(group)
    return out


# ==== reranker: dynamic micro-batching =====================================
@_gpu_work
@torch.inference_mode()
def _rr_score(pairs: List[List[str]]) -> List[float]:
    enc = _rr_tok(pairs, truncation=True, max_length=RR_MAXLEN)
    scores: List[float] = [0.0] * len(pairs)
    for g in _groups([len(x) for x in enc["input_ids"]], RR_TOKEN_BUDGET):
        inp = _rr_tok.pad({k: [enc[k][i] for i in g] for k in enc.keys()}, return_tensors="pt").to(DEVICE)
        for i, s in zip(g, _rr_model(**inp).logits.view(-1).float().cpu().tolist()):
            scores[i] = s
    return scores


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
        try:
            for i in range(0, len(flat), RR_MAXBATCH):
                scores.extend(await loop.run_in_executor(_RR_POOL, _rr_score, flat[i:i + RR_MAXBATCH]))
        except Exception as e:  # noqa: BLE001 - this batch's requests fail; the batcher carries on
            print(f"[emb-srv] rerank batch failed: {type(e).__name__}: {e}", flush=True)
            for j in jobs:
                if not j.future.done():
                    j.future.set_exception(e)
            continue
        for j, a, b in spans:
            if not j.future.done():
                j.future.set_result(scores[a:b])


# ==== embeddings ===========================================================
@_gpu_work
@torch.inference_mode()
def _embed(texts: List[str], dense: bool, sparse: bool):
    """Dense vectors and/or sparse weights, in the texts' order: run in groups within EM_TOKEN_BUDGET (_groups)."""
    enc = _em_tok(texts, truncation=True, max_length=EM_MAXLEN)
    vectors = [None] * len(texts) if dense else None
    sparse_out = [None] * len(texts) if sparse else None
    for g in _groups([len(x) for x in enc["input_ids"]], EM_TOKEN_BUDGET):
        inp = _em_tok.pad({k: [enc[k][i] for i in g] for k in enc.keys()}, return_tensors="pt").to(EM_DEVICE)
        v, sp = _embed_batch(inp, dense, sparse)
        for j, i in enumerate(g):
            if dense:
                vectors[i] = v[j]
            if sparse:
                sparse_out[i] = sp[j]
    return vectors, sparse_out


def _embed_batch(inp, dense: bool, sparse: bool):
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
    watch = asyncio.create_task(_gpu_watch())
    yield
    task.cancel()
    watch.cancel()


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
    vectors, sparse = await asyncio.get_running_loop().run_in_executor(_EM_POOL, _embed, req.texts, req.dense, req.sparse)
    resp = {}
    if req.dense:
        resp["vectors"] = vectors
    if req.sparse:
        resp["sparse"] = sparse
    return resp


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()


def admin(authorization: Optional[str] = Header(default=None)) -> None:
    if ADMIN_TOKEN and authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


class GpuMemory(BaseModel):
    release_when_idle: Optional[bool] = None
    idle_seconds: Optional[int] = None


@app.get("/admin/gpu-memory", dependencies=[Depends(admin)])
def gpu_memory_state():
    return _gpu_state()


@app.put("/admin/gpu-memory", dependencies=[Depends(admin)])
def gpu_memory_set(body: GpuMemory):
    if body.release_when_idle is not None:
        _GPU["release_when_idle"] = bool(body.release_when_idle)
    if body.idle_seconds is not None:
        _GPU["idle_seconds"] = max(10, int(body.idle_seconds))
    print(f"[emb-srv] gpu memory: release when idle {_GPU['release_when_idle']}, after {_GPU['idle_seconds']} s", flush=True)
    return _gpu_state()


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "rerank_model": RR_PATH,
            "rerank_max_batch": RR_MAXBATCH, "embed_enabled": EMBED_ENABLED,
            "embed_model": EM_PATH if EMBED_ENABLED else None,
            "embed_device": EM_DEVICE if EMBED_ENABLED else None}
