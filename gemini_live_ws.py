"""
gemini_live_ws.py — full-duplex voice bridge between the browser
(mic in / speaker out) and the real Gemini Live API. Ported from
Ashmit0107/Health-Companion's backend/gemini_live_ws.py (audited at
C:\\SIH\\health_companion\\backend\\gemini_live_ws.py), retargeted for
Qti pi's combined cardiac + neuro/seizure risk vertical.

This is a genuine live streaming session, NOT the separate
STT -> text-LLM -> TTS pipeline used elsewhere in this project's
/assistant/query (that one stays as the offline/no-API-key fallback).

Protocol over the WebSocket (/ws/voice):
  Browser -> Server:
    - binary frames: raw PCM16 mono audio @ 16000 Hz (one frame per chunk)
    - text frame '{"type":"end_turn"}': optional, signals user paused speaking
      (Gemini's own VAD also handles turn-taking automatically)

  Server -> Browser:
    - binary frames: raw PCM16 mono audio @ 24000 Hz (Gemini's output rate) —
      play these back-to-back on the client
    - text frame '{"type":"transcript","role":"user"|"model","text":"..."}':
      live transcript of either side, for the chat log UI
    - text frame '{"type":"error","message":"..."}': fatal error, then closes
    - text frame '{"type":"turn_complete"}': model finished speaking this turn

Requires GEMINI_API_KEY in backend/.env. If it's not set, the WebSocket
accepts the connection, immediately sends a type=error frame explaining
that, and closes — the frontend should fall back to the existing text
/assistant/query path in that case.
"""
import asyncio
import json
import logging
import os
import traceback

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("gemini_live_ws")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# Gemini Live model name. Override via GEMINI_LIVE_MODEL in backend/.env if
# Google renames/retires this model string — everything else in this file is
# generic and doesn't need to change.
LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")

# If LIVE_MODEL isn't found (Google retires/renames Live models often), try
# these next, in order, before giving up.
_FALLBACK_LIVE_MODELS = [
    m for m in [
        "gemini-3.1-flash-live-preview",
        "gemini-live-2.5-flash-preview-native-audio-09-2025",
        "gemini-live-2.5-flash-preview",
    ]
    if m != LIVE_MODEL
]

VOICE_SYSTEM_INSTRUCTION = (
    "You are the Qti pi voice assistant, embedded in a personal safety "
    "wearable app for people in India that tracks both cardiac and "
    "neurological (ear-EEG) health. Detect the language the user is "
    "speaking and always reply in that same language (fully multilingual — "
    "Hindi, English, Bengali, Tamil, Telugu, Marathi, and any other language "
    "the user speaks). Explain the user's vitals, cardiac risk, seizure/tremor "
    "risk, and disaster (heat/flood/pollution/cyclone) alerts in plain, "
    "reassuring, spoken-friendly language. You are not a doctor — for anything "
    "serious, tell the user to seek medical help or press the SOS button. Keep "
    "spoken replies short: 1-3 sentences, conversational."
)


def _build_context_instruction(context: dict) -> str:
    return (
        f"{VOICE_SYSTEM_INSTRUCTION}\n\n"
        f"Current risk/alerts context (JSON, for grounding your answers only, "
        f"do not read this JSON aloud): {json.dumps(context)}"
    )


async def handle_voice_ws(websocket: WebSocket, context: dict):
    """Main entrypoint, called from main.py's @app.websocket('/ws/voice')."""
    await websocket.accept()

    if not GEMINI_API_KEY:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": (
                "GEMINI_API_KEY is not set on the backend. Live voice requires it — "
                "add GEMINI_API_KEY=... to backend/.env and restart the backend. "
                "Falling back to text assistant is recommended until then."
            ),
        }))
        await websocket.close()
        return

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "google-genai package not installed on the backend. Run: pip install google-genai",
        }))
        await websocket.close()
        return

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Transcription config fields changed shape across google-genai SDK
    # versions (plain dict vs typed AudioTranscriptionConfig object). Try the
    # typed form first and fall back to a bare dict so this doesn't hard-fail
    # just because of a version mismatch.
    try:
        transcription_cfg = types.AudioTranscriptionConfig()
    except Exception:
        transcription_cfg = {}

    try:
        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=_build_context_instruction(context))]
            ),
            input_audio_transcription=transcription_cfg,
            output_audio_transcription=transcription_cfg,
        )
    except Exception as e:
        logger.error("Failed to build LiveConnectConfig (likely google-genai version mismatch): %s", e)
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Backend config error building Gemini Live session (check google-genai package version): {e}",
        }))
        await websocket.close()
        return

    candidate_models = [LIVE_MODEL] + _FALLBACK_LIVE_MODELS
    session_cm = None
    last_error = None
    connected_model = None
    # Transient/overload markers worth a quick retry on the SAME model before
    # moving to the next candidate -- a bare 503 UNAVAILABLE is Gemini being
    # momentarily overloaded, not a reason to give up or switch models.
    _retry_markers = ("503", "unavailable", "429", "resource_exhausted", "overloaded", "deadline")
    for candidate in candidate_models:
        for attempt, delay in enumerate((0, 1, 2)):
            if delay:
                await asyncio.sleep(delay)
            try:
                session_cm = client.aio.live.connect(model=candidate, config=live_config)
                session = await session_cm.__aenter__()
                connected_model = candidate
                break
            except Exception as e:
                last_error = e
                session_cm = None
                is_retryable = any(m in str(e).lower() for m in _retry_markers)
                logger.error(
                    "Gemini Live model '%s' attempt %d failed (retryable=%s): %s",
                    candidate, attempt + 1, is_retryable, e,
                )
                if not is_retryable:
                    break  # move on to next model candidate immediately
                continue
        if session_cm is not None:
            break

    if session_cm is None:
        logger.error("All Gemini Live models failed: %s\n%s", last_error, traceback.format_exc())
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Failed to start Gemini Live session on any known model ({', '.join(candidate_models)}). Last error: {last_error}",
            }))
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        return

    logger.info("Gemini Live session connected using model '%s'", connected_model)
    try:
        if True:

            async def pump_browser_to_gemini():
                """Forward mic audio from the browser WebSocket into the Gemini session."""
                try:
                    while True:
                        message = await websocket.receive()
                        if message["type"] == "websocket.disconnect":
                            break
                        if "bytes" in message and message["bytes"] is not None:
                            pcm_chunk = message["bytes"]
                            await session.send_realtime_input(
                                audio=types.Blob(data=pcm_chunk, mime_type="audio/pcm;rate=16000")
                            )
                        elif "text" in message and message["text"] is not None:
                            # Optional control messages from the client (e.g. end_turn) —
                            # Gemini's built-in VAD handles turn-taking automatically,
                            # so this is currently informational only.
                            try:
                                json.loads(message["text"])
                            except json.JSONDecodeError:
                                pass
                except WebSocketDisconnect:
                    pass

            async def pump_gemini_to_browser():
                """Forward Gemini's audio + transcripts back to the browser."""
                try:
                    async for response in session.receive():
                        server_content = getattr(response, "server_content", None)
                        if server_content is None:
                            continue

                        model_turn = getattr(server_content, "model_turn", None)
                        if model_turn is not None:
                            for part in (model_turn.parts or []):
                                inline_data = getattr(part, "inline_data", None)
                                if inline_data is not None and inline_data.data:
                                    await websocket.send_bytes(inline_data.data)

                        in_transcript = getattr(server_content, "input_transcription", None)
                        if in_transcript is not None and getattr(in_transcript, "text", None):
                            await websocket.send_text(json.dumps({
                                "type": "transcript", "role": "user", "text": in_transcript.text,
                            }))

                        out_transcript = getattr(server_content, "output_transcription", None)
                        if out_transcript is not None and getattr(out_transcript, "text", None):
                            await websocket.send_text(json.dumps({
                                "type": "transcript", "role": "model", "text": out_transcript.text,
                            }))

                        if getattr(server_content, "turn_complete", False):
                            await websocket.send_text(json.dumps({"type": "turn_complete"}))
                except Exception as e:
                    logger.error("Gemini Live stream error: %s\n%s", e, traceback.format_exc())
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "error", "message": f"Gemini Live stream error: {e}",
                        }))
                    except Exception:
                        pass

            to_gemini = asyncio.create_task(pump_browser_to_gemini())
            to_browser = asyncio.create_task(pump_gemini_to_browser())
            done, pending = await asyncio.wait(
                {to_gemini, to_browser}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()

    except Exception as e:
        logger.error("Failed to start/run Gemini Live session: %s\n%s", e, traceback.format_exc())
        try:
            await websocket.send_text(json.dumps({
                "type": "error", "message": f"Failed to start Gemini Live session: {e}",
            }))
        except Exception:
            pass
    finally:
        try:
            await session_cm.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
