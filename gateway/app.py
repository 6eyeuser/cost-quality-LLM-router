"""OpenAI-compatible routing gateway.

Production behaviours
  * embedding + classifier run in a thread pool (never block the event loop)
  * one shared, pooled httpx client (created in the lifespan handler)
  * threshold comes from the training artifact (tuned on the validation split), not a constant
  * hard rules force the strong model (tools, images, very long prompts, X-Router-Force header)
  * fail-safe: any router error -> strong model; cheap-model errors -> automatic fallback to strong
  * optional cascade mode: escalate if the cheap answer is empty/truncated
  * streaming pass-through (stream=true)
  * every request logged to SQLite; optional shadow mode samples cheap answers vs. strong answers

Run:  uvicorn gateway.app:app --port 8000
"""
from __future__ import annotations

import asyncio
import hmac
import json
import math
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from router import db as dbm
from router.config import ARTIFACT_PATH, CHEAP, DB_PATH, STRONG, UPSTREAM_BASE_URL, ModelSpec, get_api_key
from router.graders import extract_number
from router.inference import LoadedRouter, load_router


@dataclass
class Settings:
    artifact_path: str = str(ARTIFACT_PATH)
    upstream_url: str = UPSTREAM_BASE_URL.rstrip("/") + "/chat/completions"
    api_key: Optional[str] = field(default_factory=get_api_key)
    cheap: ModelSpec = CHEAP
    strong: ModelSpec = STRONG
    threshold_override: Optional[float] = None
    mode: str = "route"                 # route | cascade
    shadow_rate: float = 0.0            # fraction of cheap-routed requests also sent to strong (logged only)
    log_db: str = str(DB_PATH)
    workers: int = 4
    max_cheap_chars: int = 12000        # longer prompts exceed what we trust the cheap model with
    max_body_bytes: int = 1_000_000     # reject larger request bodies (413) before any parsing/embedding
    gateway_api_key: Optional[str] = None   # if set, clients must send "Authorization: Bearer <key>"
    allow_force: bool = False           # honour the X-Router-Force header (off by default: it bypasses routing)
    uncertain_band: float = 0.0         # p in [thr, thr+band) is "uncertain": cheap answer is verified, else escalated
    transport: Any = None               # injected in tests (httpx.MockTransport)
    router: Any = None                  # injected in tests


def settings_from_env() -> Settings:
    thr = os.environ.get("ROUTER_THRESHOLD")
    return Settings(
        threshold_override=float(thr) if thr else None,
        mode=os.environ.get("ROUTER_MODE", "route"),
        shadow_rate=float(os.environ.get("SHADOW_RATE", "0")),
        workers=int(os.environ.get("ROUTER_WORKERS", "4")),
        log_db=os.environ.get("ROUTER_LOG_DB", str(DB_PATH)),
        max_cheap_chars=int(os.environ.get("MAX_CHEAP_CHARS", "12000")),
        max_body_bytes=int(os.environ.get("MAX_BODY_BYTES", "1000000")),
        gateway_api_key=os.environ.get("GATEWAY_API_KEY") or None,
        allow_force=os.environ.get("ROUTER_ALLOW_FORCE", "0") == "1",
        uncertain_band=float(os.environ.get("ROUTER_UNCERTAIN_BAND", "0")),
    )


# --------------------------------------------------------------------------------------------
class RequestLogger:
    def __init__(self, path: str):
        import threading
        self.conn = dbm.init_db(path)
        self.conn.close()
        self.conn = dbm.connect(path, check_same_thread=False)
        self.lock = threading.Lock()

    def log(self, **row):
        cols = ",".join(row)
        with self.lock:
            self.conn.execute(f"INSERT INTO request_log ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
            self.conn.commit()

    def log_shadow(self, **row):
        cols = ",".join(row)
        with self.lock:
            self.conn.execute(f"INSERT INTO shadow_log ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
            self.conn.commit()

    def stats(self) -> dict:
        with self.lock:
            r = self.conn.execute(
                "SELECT COUNT(*), SUM(selected_model=?), AVG(overhead_ms), SUM(fallback), SUM(escalated), "
                "SUM(cost_usd), SUM(est_saved_usd) FROM request_log", (CHEAP.name,)).fetchone()
            sh = self.conn.execute("SELECT COUNT(*), AVG(agree) FROM shadow_log").fetchone()
        n = r[0] or 0
        return {"requests": n, "pct_cheap": (r[1] or 0) / n if n else None, "avg_overhead_ms": r[2],
                "fallbacks": r[3] or 0, "escalations": r[4] or 0, "spent_usd": r[5] or 0.0,
                "est_saved_usd": r[6] or 0.0, "shadow_samples": sh[0], "shadow_agreement": sh[1]}


# --------------------------------------------------------------------------------------------
def extract_prompt(messages: list[dict]) -> tuple[str, bool]:
    """Text of the last user message, and whether any message contains an image."""
    has_image, text = False, ""
    for m in messages:
        if isinstance(m.get("content"), list) and any(p.get("type") != "text" for p in m["content"] if isinstance(p, dict)):
            has_image = True
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):
                c = "\n".join(str(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
            text = c if isinstance(c, str) else ""
            break
    return text, has_image


@dataclass
class Decision:
    model: ModelSpec
    reason: str
    p: Optional[float]
    overhead_ms: float
    uncertain: bool = False


def usage_cost(spec: ModelSpec, body: dict) -> tuple[int, int, float]:
    u = (body or {}).get("usage") or {}
    pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    return pt, ct, spec.cost(pt, ct)


def cheap_answer_ok(body: dict) -> bool:
    """Free quality checks used by cascade mode: non-empty and not truncated."""
    try:
        ch = body["choices"][0]
        return bool((ch["message"].get("content") or "").strip()) and ch.get("finish_reason") != "length"
    except (KeyError, IndexError, TypeError):
        return False


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or settings_from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        router: LoadedRouter = cfg.router or load_router(cfg.artifact_path)
        router.warmup()
        app.state.router = router
        app.state.threshold = cfg.threshold_override if cfg.threshold_override is not None else router.threshold
        app.state.executor = ThreadPoolExecutor(max_workers=cfg.workers, thread_name_prefix="router")
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0), transport=cfg.transport,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))
        app.state.logger = RequestLogger(cfg.log_db)
        app.state.bg: set = set()
        yield
        await app.state.client.aclose()
        app.state.executor.shutdown(wait=False)

    app = FastAPI(title="Cost-Quality LLM Router", lifespan=lifespan)

    def spawn(coro):
        t = asyncio.create_task(coro)
        app.state.bg.add(t)
        t.add_done_callback(app.state.bg.discard)

    def log_async(**row):
        fut = asyncio.get_running_loop().run_in_executor(None, lambda: app.state.logger.log(**row))
        app.state.bg.add(fut)
        fut.add_done_callback(app.state.bg.discard)

    # ---- routing decision ------------------------------------------------------------------
    async def decide(payload: dict, headers) -> Decision:
        t0 = time.perf_counter()
        thr = app.state.threshold

        def done(model, reason, p=None, uncertain=False):
            return Decision(model, reason, p, (time.perf_counter() - t0) * 1000, uncertain)

        force = (headers.get("x-router-force") or "").lower() if cfg.allow_force else ""
        if force in ("cheap", "strong"):
            return done(cfg.cheap if force == "cheap" else cfg.strong, "forced")
        prompt, has_image = extract_prompt(payload.get("messages", []))
        if not prompt:
            raise HTTPException(400, "No user message found.")
        if payload.get("tools") or payload.get("functions") or payload.get("response_format"):
            return done(cfg.strong, "rule:structured_output_or_tools")
        if has_image:
            return done(cfg.strong, "rule:multimodal")
        if len(prompt) > cfg.max_cheap_chars:
            return done(cfg.strong, "rule:long_prompt")
        if math.isinf(thr):
            return done(cfg.strong, "ml:target_unreachable")
        try:
            p = await asyncio.get_running_loop().run_in_executor(app.state.executor, app.state.router.p_cheap_one, prompt)
        except Exception:  # fail safe: if the router breaks, quality wins
            return done(cfg.strong, "fail_safe:router_error")
        if p < thr:
            return done(cfg.strong, "ml", p)
        uncertain = p < thr + cfg.uncertain_band
        if uncertain and payload.get("stream"):
            return done(cfg.strong, "ml:uncertain_stream", p)    # a streamed answer can't be verified, so don't risk it
        return done(cfg.cheap, "ml:uncertain" if uncertain else "ml", p, uncertain)

    def check_auth(request: Request) -> None:
        if cfg.gateway_api_key is None:
            return
        given = (request.headers.get("authorization") or "")
        token = given[7:] if given.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(token.encode(), cfg.gateway_api_key.encode()):
            raise HTTPException(401, "Invalid or missing gateway API key.")

    async def read_payload(request: Request) -> dict:
        check_auth(request)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > cfg.max_body_bytes:
            raise HTTPException(413, f"Request body exceeds {cfg.max_body_bytes} bytes.")
        raw = await request.body()
        if len(raw) > cfg.max_body_bytes:
            raise HTTPException(413, f"Request body exceeds {cfg.max_body_bytes} bytes.")
        try:
            payload = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "Body is not valid JSON.")
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            raise HTTPException(400, "Body must be a JSON object with a 'messages' list.")
        if any(not isinstance(m, dict) for m in payload["messages"]):
            raise HTTPException(400, "Every message must be an object.")
        return payload

    def resp_headers(d: Decision, rid: str, **extra) -> dict:
        h = {"X-Router-Selected-Model": d.model.name, "X-Router-Reason": d.reason, "X-Router-Request-Id": rid,
             "X-Router-Overhead-Ms": f"{d.overhead_ms:.2f}", "X-Router-Threshold": f"{app.state.threshold:.4f}"}
        if d.p is not None:
            h["X-Router-Probability"] = f"{d.p:.4f}"
        h.update({k: str(v) for k, v in extra.items()})
        return h

    async def post(model: ModelSpec, payload: dict, stream: bool = False):
        body = dict(payload, model=model.name)
        req = app.state.client.build_request(
            "POST", cfg.upstream_url, json=body,
            headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"})
        return await app.state.client.send(req, stream=stream)

    def should_fallback(resp: Optional[httpx.Response]) -> bool:
        return resp is None or (resp.status_code != 200 and resp.status_code not in (401, 403))

    # ---- endpoints -------------------------------------------------------------------------
    @app.post("/v1/route")
    async def route_only(request: Request):
        """Decision only (no upstream call). Used to benchmark routing overhead in isolation."""
        d = await decide(await read_payload(request), request.headers)
        return JSONResponse({"model": d.model.name, "reason": d.reason, "p_cheap": d.p, "overhead_ms": d.overhead_ms},
                            headers=resp_headers(d, "-"))

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        rid = uuid.uuid4().hex[:12]
        payload = await read_payload(request)
        d = await decide(payload, request.headers)
        stream = bool(payload.get("stream"))
        prompt, _ = extract_prompt(payload.get("messages", []))
        model, fallback, escalated, spent = d.model, False, False, 0.0
        escalation_failed = False
        t_up = time.perf_counter()

        # ---------------- streaming ----------------
        if stream:
            try:
                resp = await post(model, payload, stream=True)
            except httpx.HTTPError:
                resp = None
            if model is cfg.cheap and should_fallback(resp):
                if resp is not None:
                    await resp.aclose()
                model, fallback = cfg.strong, True
                try:
                    resp = await post(model, payload, stream=True)
                except httpx.HTTPError:
                    resp = None
            if resp is None:
                raise HTTPException(502, "Upstream unreachable.")
            if resp.status_code != 200:
                err = await resp.aread()
                await resp.aclose()
                return JSONResponse(_json_or_text(err), status_code=resp.status_code, headers=resp_headers(d, rid))

            async def gen():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()

            log_async(request_id=rid, selected_model=model.name, reason=d.reason, p_cheap=d.p, threshold=app.state.threshold,
                      overhead_ms=d.overhead_ms, upstream_ms=None, status=200, fallback=int(fallback), escalated=0,
                      stream=1, prompt_chars=len(prompt), prompt_tokens=None, completion_tokens=None,
                      cost_usd=None, est_saved_usd=None)
            return StreamingResponse(gen(), media_type="text/event-stream",
                                     headers=resp_headers(d, rid, **{"X-Router-Fallback": int(fallback),
                                                                     "X-Router-Final-Model": model.name}))

        # ---------------- non-streaming ----------------
        try:
            resp = await post(model, payload)
        except httpx.HTTPError:
            resp = None
        if model is cfg.cheap and should_fallback(resp):
            model, fallback = cfg.strong, True
            try:
                resp = await post(model, payload)
            except httpx.HTTPError:
                resp = None
        if resp is None:
            raise HTTPException(502, "Upstream unreachable.")
        body = _safe_json(resp)
        if model is cfg.cheap:
            spent += usage_cost(cfg.cheap, body)[2]
            if (cfg.mode == "cascade" or d.uncertain) and resp.status_code == 200 and not cheap_answer_ok(body):
                try:
                    strong_resp = await post(cfg.strong, payload)
                except httpx.HTTPError:
                    strong_resp = None
                if strong_resp is not None and strong_resp.status_code == 200:
                    escalated, model, resp = True, cfg.strong, strong_resp
                    body = _safe_json(resp)
                    spent += usage_cost(cfg.strong, body)[2]
                else:
                    escalation_failed = True   # keep the (imperfect) cheap answer rather than failing the request
        else:
            spent += usage_cost(cfg.strong, body)[2]
        upstream_ms = (time.perf_counter() - t_up) * 1000
        pt, ct, _ = usage_cost(model, body)
        saved = (cfg.strong.cost(pt, ct) - cfg.cheap.cost(pt, ct)) if model is cfg.cheap else 0.0

        log_async(request_id=rid, selected_model=model.name, reason=d.reason, p_cheap=d.p, threshold=app.state.threshold,
                  overhead_ms=d.overhead_ms, upstream_ms=upstream_ms, status=resp.status_code, fallback=int(fallback),
                  escalated=int(escalated), stream=0, prompt_chars=len(prompt), prompt_tokens=pt, completion_tokens=ct,
                  cost_usd=spent, est_saved_usd=saved)

        if (cfg.shadow_rate > 0 and model is cfg.cheap and resp.status_code == 200 and random.random() < cfg.shadow_rate):
            spawn(shadow(payload, prompt, d.p, rid, body))

        if resp.status_code != 200:
            return JSONResponse(body, status_code=resp.status_code, headers=resp_headers(d, rid))
        return JSONResponse(body, headers=resp_headers(
            d, rid, **{"X-Router-Fallback": int(fallback), "X-Router-Escalated": int(escalated),
                       "X-Router-Escalation-Failed": int(escalation_failed), "X-Router-Final-Model": model.name}))

    async def shadow(payload: dict, prompt: str, p: Optional[float], rid: str, cheap_body: dict):
        """Drift monitoring: also ask the strong model, log both answers + a cheap agreement signal."""
        try:
            sresp = await post(cfg.strong, {**payload, "stream": False})
            sbody = _safe_json(sresp)
            ctext = (cheap_body["choices"][0]["message"].get("content") or "")
            stext = (sbody["choices"][0]["message"].get("content") or "")
            a, b = extract_number(ctext), extract_number(stext)
            agree = float(a is not None and b is not None and abs(a - b) < 1e-6) if (a is not None or b is not None) else None
            await asyncio.get_running_loop().run_in_executor(None, lambda: app.state.logger.log_shadow(
                request_id=rid, prompt=prompt[:4000], p_cheap=p, cheap_response=ctext, strong_response=stext,
                cheap_cost=usage_cost(cfg.cheap, cheap_body)[2], strong_cost=usage_cost(cfg.strong, sbody)[2], agree=agree))
        except Exception:
            pass  # shadow traffic must never affect users

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "router": app.state.router.name, "threshold": app.state.threshold, "mode": cfg.mode,
                "cheap": cfg.cheap.name, "strong": cfg.strong.name}

    @app.get("/stats")
    async def stats(request: Request):
        check_auth(request)
        return app.state.logger.stats()

    return app


def _safe_json(resp: Optional[httpx.Response]) -> dict:
    if resp is None:
        return {"error": {"message": "upstream unreachable"}}
    try:
        return resp.json()
    except ValueError:
        return {"error": {"message": resp.text[:500]}}


def _json_or_text(raw: bytes):
    try:
        return json.loads(raw)
    except ValueError:
        return {"error": {"message": raw.decode("utf-8", "ignore")[:500]}}


app = create_app()
