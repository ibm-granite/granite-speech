from __future__ import annotations

import argparse
import asyncio
import io
import sys
import uuid
import wave
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np

import granite_speech
from granite_speech.errors import GraniteSpeechError


DEFAULT_MODEL = "granite-speech-4.1-2b"
DEFAULT_SAMPLE_RATE = 16000


class MissingPipecatDependency(RuntimeError):
    pass


def _import_pipecat() -> dict[str, Any]:
    try:
        import uvicorn
        from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
        from fastapi.responses import RedirectResponse
        from loguru import logger
        from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker
        from pipecat.processors.audio.vad_processor import VADProcessor
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
        from pipecat.services.stt_service import SegmentedSTTService
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
        from pipecat.transports.smallwebrtc.request_handler import (
            SmallWebRTCPatchRequest,
            SmallWebRTCRequest,
            SmallWebRTCRequestHandler,
        )
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
        from pipecat.utils.time import time_now_iso8601
        from pipecat.workers.runner import WorkerRunner
    except ImportError as exc:
        raise MissingPipecatDependency(
            "Install the example dependencies with `cd examples && uv sync`."
        ) from exc

    return {
        "BackgroundTasks": BackgroundTasks,
        "ErrorFrame": ErrorFrame,
        "FastAPI": FastAPI,
        "Frame": Frame,
        "FrameDirection": FrameDirection,
        "FrameProcessor": FrameProcessor,
        "IceServer": IceServer,
        "HTTPException": HTTPException,
        "Pipeline": Pipeline,
        "PipelineParams": PipelineParams,
        "PipelineWorker": PipelineWorker,
        "RedirectResponse": RedirectResponse,
        "Request": Request,
        "SegmentedSTTService": SegmentedSTTService,
        "SileroVADAnalyzer": SileroVADAnalyzer,
        "SmallWebRTCPatchRequest": SmallWebRTCPatchRequest,
        "SmallWebRTCConnection": SmallWebRTCConnection,
        "SmallWebRTCPrebuiltUI": SmallWebRTCPrebuiltUI,
        "SmallWebRTCRequest": SmallWebRTCRequest,
        "SmallWebRTCRequestHandler": SmallWebRTCRequestHandler,
        "SmallWebRTCTransport": SmallWebRTCTransport,
        "TranscriptionFrame": TranscriptionFrame,
        "TransportParams": TransportParams,
        "VADParams": VADParams,
        "VADProcessor": VADProcessor,
        "WorkerRunner": WorkerRunner,
        "logger": logger,
        "time_now_iso8601": time_now_iso8601,
        "uvicorn": uvicorn,
    }


def _pcm_wav_to_float32_mono(audio: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM audio from Pipecat, got {sample_width * 8}-bit")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(samples, dtype=np.float32), sample_rate


def build_granite_stt_service(pipecat: dict[str, Any]):
    SegmentedSTTService = pipecat["SegmentedSTTService"]
    TranscriptionFrame = pipecat["TranscriptionFrame"]
    ErrorFrame = pipecat["ErrorFrame"]
    time_now_iso8601 = pipecat["time_now_iso8601"]

    class GraniteSpeechSTTService(SegmentedSTTService):
        def __init__(
            self,
            *,
            model_name: str,
            device: str | None,
            download_root: str | None,
            revision: str | None,
            local_files_only: bool,
            language: str | None,
            max_new_tokens: int,
            chunk_length: float,
        ):
            super().__init__(
                audio_passthrough=False,
                sample_rate=DEFAULT_SAMPLE_RATE,
            )
            self._model = granite_speech.load_model(
                model_name,
                device=device,
                download_root=download_root,
                revision=revision,
                local_files_only=local_files_only,
            )
            self._language = language
            self._max_new_tokens = max_new_tokens
            self._chunk_length = chunk_length

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Any, None]:
            try:
                wav, sample_rate = _pcm_wav_to_float32_mono(audio)
                if wav.size == 0:
                    return

                result = await asyncio.to_thread(
                    self._model.transcribe,
                    wav,
                    sample_rate=sample_rate,
                    language=self._language,
                    max_new_tokens=self._max_new_tokens,
                    chunk_length=self._chunk_length,
                    chunk_overlap=0.0,
                    verbose=False,
                )
            except GraniteSpeechError as exc:
                yield ErrorFrame(error=f"Granite Speech transcription failed: {exc}")
                return
            except Exception as exc:
                yield ErrorFrame(error=f"Unexpected Granite Speech error: {exc}", exception=exc)
                return

            text = result["text"].strip()
            if not text:
                return

            yield TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=None,
                result=result,
            )

    return GraniteSpeechSTTService


def build_transcription_logger(pipecat: dict[str, Any]):
    Frame = pipecat["Frame"]
    FrameDirection = pipecat["FrameDirection"]
    FrameProcessor = pipecat["FrameProcessor"]
    TranscriptionFrame = pipecat["TranscriptionFrame"]
    ErrorFrame = pipecat["ErrorFrame"]

    class TranscriptionLogger(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, TranscriptionFrame):
                print(f"Transcription: {frame.text}", flush=True)
                result = frame.result or {}
                warnings = result.get("warnings", [])
                if warnings:
                    print(f"Warnings: {len(warnings)}", file=sys.stderr, flush=True)
            elif isinstance(frame, ErrorFrame):
                print(f"Error: {frame.error}", file=sys.stderr, flush=True)

            await self.push_frame(frame, direction)

    return TranscriptionLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a Pipecat WebRTC UI backed by Granite Speech transcription."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="device override, e.g. cuda, mps, cpu")
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--language", default=None, help="optional source-language hint, e.g. en")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--chunk-length", type=float, default=30.0)
    parser.add_argument("--vad-start-secs", type=float, default=0.2)
    parser.add_argument("--vad-stop-secs", type=float, default=0.6)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def create_app(args: argparse.Namespace, pipecat: dict[str, Any]):
    logger = pipecat["logger"]
    BackgroundTasks = pipecat["BackgroundTasks"]
    FastAPI = pipecat["FastAPI"]
    HTTPException = pipecat["HTTPException"]
    RedirectResponse = pipecat["RedirectResponse"]
    Request = pipecat["Request"]
    SmallWebRTCPrebuiltUI = pipecat["SmallWebRTCPrebuiltUI"]
    IceServer = pipecat["IceServer"]
    SmallWebRTCPatchRequest = pipecat["SmallWebRTCPatchRequest"]
    SmallWebRTCRequest = pipecat["SmallWebRTCRequest"]
    SmallWebRTCRequestHandler = pipecat["SmallWebRTCRequestHandler"]

    active_sessions: dict[str, Any] = {}
    ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]
    request_handler = SmallWebRTCRequestHandler(ice_servers=ice_servers)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await request_handler.close()
        active_sessions.clear()

    app = FastAPI(lifespan=lifespan)
    app.mount("/client", SmallWebRTCPrebuiltUI)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/client/")

    async def start(request: Request):
        try:
            request_data = await request.json()
        except Exception:
            request_data = {}
        if not isinstance(request_data, dict):
            request_data = {}

        transport = request_data.get("transport", "webrtc")
        if transport != "webrtc":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported transport '{transport}'. This example only supports WebRTC.",
            )

        session_id = str(uuid.uuid4())
        active_sessions[session_id] = request_data.get("body", {})

        response: dict[str, Any] = {"sessionId": session_id}
        if request_data.get("enableDefaultIceServers"):
            response["iceConfig"] = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        return response

    start.__annotations__["request"] = Request
    app.post("/start")(start)

    async def offer(
        request: SmallWebRTCRequest,
        background_tasks: BackgroundTasks,
        session_id: str | None = None,
    ):
        session_label = session_id or "direct"

        async def webrtc_connection_callback(connection):
            logger.info(
                f"Starting Pipecat session {session_label} for pc_id: {connection.pc_id}"
            )
            background_tasks.add_task(run_session, connection, args, pipecat)

        return await request_handler.handle_web_request(
            request=request,
            webrtc_connection_callback=webrtc_connection_callback,
        )

    offer.__annotations__["request"] = SmallWebRTCRequest
    offer.__annotations__["background_tasks"] = BackgroundTasks
    app.post("/api/offer")(offer)

    async def ice_candidate(request: SmallWebRTCPatchRequest):
        await request_handler.handle_patch_request(request)
        return {"status": "success"}

    ice_candidate.__annotations__["request"] = SmallWebRTCPatchRequest
    app.patch("/api/offer")(ice_candidate)

    async def session_offer(
        session_id: str,
        request: SmallWebRTCRequest,
        background_tasks: BackgroundTasks,
    ):
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Invalid or expired session_id")
        active_session = active_sessions[session_id]
        if request.request_data is None:
            request.request_data = active_session
        return await offer(request, background_tasks, session_id=session_id)

    session_offer.__annotations__["request"] = SmallWebRTCRequest
    session_offer.__annotations__["background_tasks"] = BackgroundTasks
    app.post("/sessions/{session_id}/api/offer")(session_offer)

    async def session_ice_candidate(session_id: str, request: SmallWebRTCPatchRequest):
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Invalid or expired session_id")
        return await ice_candidate(request)

    session_ice_candidate.__annotations__["request"] = SmallWebRTCPatchRequest
    app.patch("/sessions/{session_id}/api/offer")(session_ice_candidate)

    return app


async def run_session(webrtc_connection: Any, args: argparse.Namespace, pipecat: dict[str, Any]):
    logger = pipecat["logger"]
    GraniteSpeechSTTService = build_granite_stt_service(pipecat)
    TranscriptionLogger = build_transcription_logger(pipecat)

    transport = pipecat["SmallWebRTCTransport"](
        webrtc_connection=webrtc_connection,
        params=pipecat["TransportParams"](
            audio_in_enabled=True,
            audio_in_sample_rate=DEFAULT_SAMPLE_RATE,
            audio_out_enabled=True,
            audio_out_sample_rate=DEFAULT_SAMPLE_RATE,
        ),
    )
    vad_processor = pipecat["VADProcessor"](
        vad_analyzer=pipecat["SileroVADAnalyzer"](
            sample_rate=DEFAULT_SAMPLE_RATE,
            params=pipecat["VADParams"](
                start_secs=args.vad_start_secs,
                stop_secs=args.vad_stop_secs,
            ),
        )
    )
    stt = GraniteSpeechSTTService(
        model_name=args.model,
        device=args.device,
        download_root=args.download_root,
        revision=args.revision,
        local_files_only=args.local_files_only,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        chunk_length=args.chunk_length,
    )

    pipeline = pipecat["Pipeline"](
        [
            transport.input(),
            vad_processor,
            stt,
            TranscriptionLogger(),
            transport.output(),
        ]
    )
    worker = pipecat["PipelineWorker"](
        pipeline,
        params=pipecat["PipelineParams"](
            audio_in_sample_rate=DEFAULT_SAMPLE_RATE,
            audio_out_sample_rate=DEFAULT_SAMPLE_RATE,
            enable_metrics=True,
        ),
    )

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(_rtvi):
        logger.info("Pipecat client ready.")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("Pipecat client connected.")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Pipecat client disconnected.")
        await worker.cancel()

    runner = pipecat["WorkerRunner"](handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


def main() -> int:
    args = parse_args()
    try:
        pipecat = _import_pipecat()
    except MissingPipecatDependency as exc:
        print(exc, file=sys.stderr)
        return 2

    logger = pipecat["logger"]
    logger.remove(0)
    logger.add(sys.stderr, level=args.log_level.upper())

    app = create_app(args, pipecat)
    print(f"Open http://{args.host}:{args.port} in a browser.", flush=True)
    pipecat["uvicorn"].run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
