"""DeepFilterNet3 denoising for source speech audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from config.settings import get_settings
from utils.gpu_memory import clear_gpu_cache, resolve_device
from utils.logger import logger


def _fit_length(audio: torch.Tensor, length: int) -> torch.Tensor:
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    current = audio.shape[-1]
    if current == length:
        return audio
    if current > length:
        return audio[..., :length]
    pad = length - current
    return torch.nn.functional.pad(audio, (0, pad))


def _enhance_in_chunks(
    model: torch.nn.Module,
    df_state: object,
    audio: torch.Tensor,
    chunk_seconds: float,
    overlap_seconds: float,
) -> torch.Tensor:
    from df.enhance import enhance

    sample_rate = int(df_state.sr())  # type: ignore[attr-defined]
    total = int(audio.shape[-1])
    chunk = max(1, int(chunk_seconds * sample_rate))
    overlap = int(overlap_seconds * sample_rate)
    if overlap >= chunk:
        overlap = max(chunk // 5, 0)
    if total <= chunk:
        return _fit_length(enhance(model, df_state, audio), total)

    step = max(chunk - overlap, 1)
    result = torch.zeros_like(audio)
    weight = torch.zeros(total, dtype=audio.dtype)
    start = 0
    index = 0
    while start < total:
        end = min(start + chunk, total)
        piece = audio[:, start:end]
        enhanced_piece = _fit_length(enhance(model, df_state, piece), piece.shape[-1])
        length = int(enhanced_piece.shape[-1])
        fade = torch.ones(length, dtype=audio.dtype)
        if start > 0 and overlap > 0:
            ramp_in = min(overlap, length)
            fade[:ramp_in] = torch.linspace(0, 1, ramp_in)
        if end < total and overlap > 0:
            ramp_out = min(overlap, length)
            fade[-ramp_out:] = torch.minimum(fade[-ramp_out:], torch.linspace(1, 0, ramp_out))
        result[:, start : start + length] += enhanced_piece * fade.unsqueeze(0)
        weight[start : start + length] += fade
        index += 1
        if index % 10 == 0:
            logger.info("DeepFilterNet processed {} / {} samples", min(end, total), total)
        if index % 20 == 0:
            clear_gpu_cache(log=False)
        if end >= total:
            break
        start += step
    weight = weight.clamp_min(1e-6)
    return result / weight.unsqueeze(0)


def clean_audio_deepfilternet(input_wav_path: str) -> str:
    """Enhance a 16 kHz wav with DeepFilterNet3 and return the cleaned wav path."""
    source = Path(input_wav_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input wav not found: {source}")

    resolve_device()
    settings = get_settings()
    settings.ensure_output_dirs()
    output_path = settings.AUDIO_OUTPUT_DIR / f"{source.stem}_cleaned.wav"
    if output_path.is_file() and output_path.stat().st_size > 0:
        logger.info("Reusing cleaned audio {}", output_path.resolve())
        return str(output_path.resolve())

    try:
        from df.enhance import init_df
    except ImportError as exc:
        logger.warning(
            "deepfilternet is not installed ({}); skipping denoise and using raw audio: {}",
            exc,
            source.resolve(),
        )
        return str(source.resolve())

    logger.info("Loading DeepFilterNet3")
    model = None
    df_state = None
    try:
        try:
            model, df_state, _ = init_df(
                default_model="DeepFilterNet3",
                post_filter=True,
                log_level="ERROR",
                log_file=None,
            )
        except TypeError:
            model, df_state, _ = init_df(default_model="DeepFilterNet3", post_filter=True)
        samples, file_sr = sf.read(str(source), dtype="float32", always_2d=True)
        audio = torch.from_numpy(np.ascontiguousarray(samples.T))
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        model_sr = int(df_state.sr())
        if file_sr != model_sr:
            audio = torchaudio_resample(audio, file_sr, model_sr)
        logger.info(
            "Denoising {:.1f} minutes of audio in {:.0f}s chunks",
            audio.shape[-1] / model_sr / 60.0,
            settings.DEEPFILTER_CHUNK_SECONDS,
        )
        enhanced = _enhance_in_chunks(
            model,
            df_state,
            audio,
            settings.DEEPFILTER_CHUNK_SECONDS,
            settings.DEEPFILTER_OVERLAP_SECONDS,
        )
        if enhanced.shape[0] > 1:
            enhanced = enhanced.mean(dim=0, keepdim=True)
        target_sr = settings.TARGET_SAMPLE_RATE
        if model_sr != target_sr:
            enhanced = torchaudio_resample(enhanced, model_sr, target_sr)
        mono = enhanced.squeeze(0).detach().cpu().numpy()
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if peak > 1.0:
            mono = mono / peak
        sf.write(str(output_path), mono, target_sr, subtype="PCM_16")
    finally:
        if model is not None:
            del model
        if df_state is not None:
            del df_state
        clear_gpu_cache()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"DeepFilterNet did not write {output_path}")
    logger.info("Cleaned audio saved to {}", output_path.resolve())
    return str(output_path.resolve())


def torchaudio_resample(audio: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    """Resample a ``[channels, time]`` tensor without torchcodec."""
    import torchaudio

    return torchaudio.functional.resample(audio.cpu(), source_sr, target_sr)
