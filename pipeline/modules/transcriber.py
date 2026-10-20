"""WhisperX transcription, forced alignment, and Pyannote 3.1 diarization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from config.settings import get_settings
from utils.gpu_memory import clear_gpu_cache, resolve_device
from utils.logger import logger

import torch

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _log_cuda_vram(stage: str) -> None:
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        logger.info(
            "CUDA VRAM after {}: {:.2f} GB allocated, {:.2f} GB reserved",
            stage,
            allocated_gb,
            reserved_gb,
        )


def _pyannote_checkpoint_safe_globals() -> list[Any]:
    """Types embedded in trusted Hugging Face pyannote checkpoints (PyTorch weights_only)."""
    safe: list[Any] = []
    try:
        from torch.torch_version import TorchVersion

        safe.append(TorchVersion)
    except Exception:
        pass
    try:
        from pyannote.audio.core.task import Problem, Resolution, Specifications

        safe.extend([Specifications, Problem, Resolution])
    except Exception:
        pass
    return safe


try:
    registered = _pyannote_checkpoint_safe_globals()
    if registered:
        torch.serialization.add_safe_globals(registered)
except Exception:
    pass


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    score: float
    speaker: str


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str
    words: list[WordTimestamp] = Field(default_factory=list)


class LectureTranscript(BaseModel):
    source_url: str
    audio_clean_path: str
    duration_seconds: float
    segments: list[TranscriptSegment]


def _disable_hf_symlinks() -> None:
    """Windows without Developer Mode cannot create the cache symlinks Hugging Face expects."""
    import huggingface_hub.file_download as file_download

    file_download.are_symlinks_supported = lambda cache_dir=None: False  # type: ignore[assignment]


def _seconds(value: float) -> float:
    return round(float(value), 3)


def segments_from_aligned(aligned: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a WhisperX aligned result into the pipeline segment schema."""
    segments: list[dict[str, Any]] = []
    for raw in aligned.get("segments") or []:
        words_out: list[dict[str, Any]] = []
        for word in raw.get("words") or []:
            token = str(word.get("word") or "").strip()
            if not token or "start" not in word:
                continue
            start = _seconds(word["start"])
            end = _seconds(word.get("end", start))
            if end < start:
                end = start
            score = word.get("score", word.get("probability", 0.0))
            words_out.append(
                {
                    "word": token,
                    "start": start,
                    "end": end,
                    "score": round(float(score or 0.0), 4),
                    "speaker": str(word.get("speaker") or raw.get("speaker") or "UNKNOWN"),
                }
            )
        text = str(raw.get("text") or "").strip()
        if not text and words_out:
            text = " ".join(item["word"] for item in words_out)
        if not text and not words_out:
            continue
        start = _seconds(raw.get("start", words_out[0]["start"] if words_out else 0.0))
        end = _seconds(raw.get("end", words_out[-1]["end"] if words_out else start))
        if end < start:
            end = start
        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": str(raw.get("speaker") or (words_out[0]["speaker"] if words_out else "UNKNOWN")),
                "words": words_out,
            }
        )
    return segments


def _install_onset_offset(pipeline: Any, onset: float, offset: float) -> None:
    """Binarize segmentation scores with onset/offset hysteresis before clustering."""
    from pyannote.audio.utils.signal import binarize

    if getattr(pipeline, "_onset_offset_installed", False):
        return
    original_get_segmentations = pipeline.get_segmentations

    def get_segmentations(file: Any, hook: Any = None) -> Any:
        scores = original_get_segmentations(file, hook=hook)
        return binarize(scores, onset=onset, offset=offset, initial_state=False)

    pipeline.get_segmentations = get_segmentations
    pipeline._onset_offset_installed = True
    segmentation = getattr(pipeline, "segmentation", None)
    if segmentation is not None and hasattr(segmentation, "threshold"):
        try:
            segmentation.threshold = onset
        except (AttributeError, TypeError):
            logger.debug("Segmentation model has no writable threshold")
    logger.info("Pyannote turn-taking hysteresis set to onset={} offset={}", onset, offset)


def _load_diarization_pipeline(model_name: str, token: str, device: str) -> Any | None:
    """Load speaker-diarization-3.1 when the token can access that gated checkpoint."""
    import pickle

    from pyannote.audio import Pipeline

    def _from_pretrained() -> Any:
        try:
            return Pipeline.from_pretrained(model_name, use_auth_token=token)
        except TypeError:
            return Pipeline.from_pretrained(model_name, token=token)

    pipeline = None
    try:
        pipeline = _from_pretrained()
    except (pickle.UnpicklingError, RuntimeError) as exc:
        message = str(exc)
        if "Weights only load failed" not in message and "Unsupported global" not in message:
            raise
        try:
            safe = _pyannote_checkpoint_safe_globals()
            if safe and hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals(safe):
                    pipeline = _from_pretrained()
            else:
                pipeline = _from_pretrained()
        except Exception:
            raise exc

    if pipeline is None:
        return None
    pipeline.to(torch.device(device))
    return pipeline


def _speech_regions(audio: Any, onset: float, offset: float) -> list[tuple[float, float]]:
    """Turn Silero frame probabilities into speech regions with onset/offset hysteresis."""
    import torch

    vad_model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_model = vad_model.to(device)
    vad_model.eval()
    waveform = torch.from_numpy(audio)
    window = 512
    if waveform.numel() < window:
        return []
    probabilities: list[float] = []
    with torch.no_grad():
        for start in range(0, int(waveform.shape[0]) - window + 1, window):
            chunk = waveform[start : start + window].to(device)
            probabilities.append(float(vad_model(chunk, 16000).item()))
    frame = window / 16000.0
    regions: list[tuple[float, float]] = []
    active = False
    region_start = 0.0
    for index, probability in enumerate(probabilities):
        timestamp = index * frame
        if not active and probability >= onset:
            active = True
            region_start = timestamp
        elif active and probability < offset:
            active = False
            region_end = timestamp + frame
            if region_end - region_start >= 0.4:
                regions.append((region_start, region_end))
    if active:
        regions.append((region_start, len(audio) / 16000.0))
    del vad_model
    clear_gpu_cache(log=False)
    return regions


def _diarize_with_open_weights(audio: Any, onset: float, offset: float) -> Any:
    """Cluster WeSpeaker embeddings over hysteresis speech regions.

    Used when the gated speaker-diarization-3.1 weights are not available to the token.
    """
    import numpy as np
    import pandas as pd
    import torch
    from pyannote.audio.pipelines.speaker_verification import (
        ONNXWeSpeakerPretrainedSpeakerEmbedding,
    )
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    logger.warning(
        "Gated pyannote/speaker-diarization-3.1 weights are not available to this token. "
        "Diarizing with Silero speech probabilities (onset={} offset={}) and public WeSpeaker embeddings.",
        onset,
        offset,
    )
    regions = _speech_regions(audio, onset, offset)
    embedder = ONNXWeSpeakerPretrainedSpeakerEmbedding(
        "hbredin/wespeaker-voxceleb-resnet34-LM",
        device=torch.device("cpu"),
    )
    sample_rate = 16000
    minimum = max(int(embedder.min_num_samples), int(0.4 * sample_rate))
    embeddings: list[Any] = []
    kept: list[tuple[float, float]] = []
    for start, end in regions:
        start_index = int(start * sample_rate)
        end_index = int(end * sample_rate)
        if end_index - start_index < minimum:
            continue
        clip = audio[start_index : min(end_index, start_index + 8 * sample_rate)]
        waveform = torch.from_numpy(clip).unsqueeze(0).unsqueeze(0)
        embedding = embedder(waveform)
        if embedding is None or np.any(np.isnan(embedding)):
            continue
        embeddings.append(np.asarray(embedding[0], dtype=np.float32))
        kept.append((start, end))
    if not kept:
        return pd.DataFrame(columns=["start", "end", "speaker"])
    if len(kept) == 1:
        labels = [1]
    else:
        distances = pdist(np.stack(embeddings), metric="cosine")
        labels = fcluster(linkage(distances, method="average"), t=0.7, criterion="distance")
    names: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    for (start, end), label in zip(kept, labels):
        key = int(label)
        if key not in names:
            names[key] = f"SPEAKER_{len(names):02d}"
        rows.append({"start": float(start), "end": float(end), "speaker": names[key]})
    logger.info("Open-weight diarization produced {} turns and {} speakers", len(rows), len(names))
    return pd.DataFrame(rows)


def _annotation_to_rows(annotation: Any) -> list[dict[str, Any]]:
    if hasattr(annotation, "speaker_diarization"):
        annotation = annotation.speaker_diarization
    if not hasattr(annotation, "itertracks"):
        raise TypeError(f"Unexpected diarization output type: {type(annotation)!r}")
    rows: list[dict[str, Any]] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        rows.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": str(speaker),
            }
        )
    return rows


def transcribe_and_diarize(
    cleaned_wav_path: str,
    *,
    language: str | None = None,
    batch_size: int | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Transcribe, force-align, and diarize a cleaned wav.

    Returns ``duration_seconds``, ``language``, and schema-ready ``segments``.
    """
    audio_path = Path(cleaned_wav_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Cleaned wav not found: {audio_path}")

    # Retain the argument for caller compatibility, but this pipeline intentionally
    # uses English to prevent WhisperX from misdetecting Sanskrit intros as Welsh.
    del language

    settings = get_settings()
    if not settings.HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is empty. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN. "
            "Accept pyannote/speaker-diarization-3.1, pyannote/segmentation-3.0, "
            "and pyannote/wespeaker-voxceleb-resnet34-LM on Hugging Face."
        )

    _disable_hf_symlinks()
    selected_device = resolve_device(device)
    if selected_device == "cpu":
        compute_type = settings.WHISPER_COMPUTE_TYPE
        if compute_type == "float16":
            compute_type = "int8"
            logger.info("CPU device selected; using compute_type=int8")
    else:
        compute_type = "float16"
        logger.info("CUDA device selected; using compute_type=float16")
    if batch_size is not None:
        selected_batch = batch_size
    elif selected_device != "cpu":
        selected_batch = max(settings.WHISPER_BATCH_SIZE, 32)
    else:
        selected_batch = settings.WHISPER_BATCH_SIZE

    try:
        import pandas as pd
        import whisperx
    except ImportError as exc:
        raise RuntimeError("whisperx is not installed. pip install -r requirements.txt") from exc

    logger.info("Loading audio {}", audio_path)
    audio = whisperx.load_audio(str(audio_path))
    duration_seconds = _seconds(len(audio) / 16000.0)
    logger.info("Audio duration {:.1f} minutes", duration_seconds / 60.0)

    logger.info("Loading WhisperX {}", settings.WHISPER_MODEL)
    asr_model = None
    try:
        load_kwargs: dict[str, Any] = {
            "device": selected_device,
            "compute_type": compute_type,
            "download_root": str(settings.MODELS_DIR / "whisper"),
            "vad_method": "silero",
            "language": "en",
            "asr_options": {"initial_prompt": settings.WHISPER_INITIAL_PROMPT},
        }
        try:
            asr_model = whisperx.load_model(settings.WHISPER_MODEL, **load_kwargs)
        except TypeError:
            load_kwargs.pop("vad_method")
            asr_model = whisperx.load_model(settings.WHISPER_MODEL, **load_kwargs)
        _log_cuda_vram("WhisperX model load")
        logger.info("Transcribing with batch size {}", selected_batch)
        transcribed = asr_model.transcribe(
            audio,
            batch_size=selected_batch,
            language="en",
        )
        _log_cuda_vram("WhisperX transcription")
    finally:
        if asr_model is not None:
            del asr_model
        clear_gpu_cache()

    detected_language = "en"
    logger.info("Transcription language {}", detected_language)

    logger.info("Loading Wav2Vec2 alignment model for en")
    align_model = None
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code="en",
            device=selected_device,
        )
        aligned = whisperx.align(
            transcribed["segments"],
            align_model,
            metadata,
            audio,
            selected_device,
            return_char_alignments=False,
        )
    finally:
        if align_model is not None:
            del align_model
        clear_gpu_cache()

    logger.info("Loading {}", settings.PYANNOTE_MODEL)
    diarization_pipeline = None
    try:
        diarization_pipeline = _load_diarization_pipeline(
            settings.PYANNOTE_MODEL,
            settings.HF_TOKEN,
            selected_device,
        )
        if diarization_pipeline is None:
            diarize_df = _diarize_with_open_weights(
                audio,
                settings.PYANNOTE_ONSET,
                settings.PYANNOTE_OFFSET,
            )
        else:
            _install_onset_offset(
                diarization_pipeline,
                settings.PYANNOTE_ONSET,
                settings.PYANNOTE_OFFSET,
            )
            if hasattr(diarization_pipeline, "segmentation_batch_size"):
                diarization_pipeline.segmentation_batch_size = settings.PYANNOTE_SEGMENTATION_BATCH_SIZE
            if hasattr(diarization_pipeline, "embedding_batch_size"):
                diarization_pipeline.embedding_batch_size = settings.PYANNOTE_SEGMENTATION_BATCH_SIZE

            import torch

            waveform = torch.from_numpy(audio[None, :])
            file = {"waveform": waveform, "sample_rate": 16000, "uri": audio_path.stem}

            def _hook(
                step_name: str,
                _artifact: Any,
                file: Any = None,
                total: Any = None,
                completed: Any = None,
            ) -> None:
                del file
                if total and completed is not None and int(total) > 0 and int(completed) % max(int(total) // 5, 1) == 0:
                    logger.info("Diarization {}: {}/{}", step_name, completed, total)

            call_kwargs: dict[str, Any] = {
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "hook": _hook,
            }
            try:
                annotation = diarization_pipeline(file, **call_kwargs)
            except TypeError:
                call_kwargs.pop("hook")
                annotation = diarization_pipeline(file, **call_kwargs)
            rows = _annotation_to_rows(annotation)
            diarize_df = pd.DataFrame(rows, columns=["start", "end", "speaker"])
            logger.info("Diarization produced {} speaker turns", len(diarize_df))
        if diarize_df.empty:
            assigned = aligned
        else:
            assigned = whisperx.assign_word_speakers(diarize_df, aligned, fill_nearest=True)
    finally:
        if diarization_pipeline is not None:
            del diarization_pipeline
        clear_gpu_cache()

    segments = segments_from_aligned(assigned)
    logger.info("Transcript contains {} segments", len(segments))
    return {
        "duration_seconds": duration_seconds,
        "language": detected_language,
        "segments": segments,
    }
