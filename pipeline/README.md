# Pipeline: ingestion, transcription, clip selection, Shorts render

Local pipeline for long-form lecture video on an NVIDIA GPU (24 GB VRAM tested during development). It downloads a YouTube video, isolates 16 kHz mono audio, denoises with DeepFilterNet3, transcribes with WhisperX, force-aligns words, labels speakers with Pyannote 3.1, selects clips with an LLM, and renders 9:16 Shorts.

## Setup

Python 3.11, ffmpeg on `PATH`, and an NVIDIA driver that can run CUDA 12.8 wheels. On Windows, use 3.11: DeepFilterLib 0.5.6 publishes a wheel for that version and not for 3.12.

```bash
cd pipeline
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
.\scripts\install_deepfilter.ps1
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Copy `.env.example` to `.env` and set API keys plus `BRAND_*` fields.

## Hugging Face token

Pyannote 3.1 is gated. Create a read token at <https://huggingface.co/settings/tokens> and accept the conditions for:

- <https://huggingface.co/pyannote/speaker-diarization-3.1>
- <https://huggingface.co/pyannote/segmentation-3.0>
- <https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM>

Export the token before running:

```bash
export HF_TOKEN=hf_your_token
```

On PowerShell: `$env:HF_TOKEN = "hf_your_token"`. The token is never written into logs or transcripts.

## YouTube cookies

If downloads return HTTP 429, export a Netscape `cookies.txt` from a logged-in browser session and place it at `pipeline/cookies.txt`, or set `COOKIES_FILE_PATH`. The downloader waits `YTDLP_SLEEP_INTERVAL` seconds (default 30) between requests and backs off exponentially after 429 responses. Set `USER_AGENT=rotate` to pick a modern Chrome user-agent per download.

## Run

```bash
python run_ingestion_pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

On Windows:

```powershell
.\run_phase1.ps1 "https://www.youtube.com/watch?v=VIDEO_ID"
```

Optional flags: `--language en`, `--batch-size 16`, `--device cuda`, `--num-speakers`, `--min-speakers`, `--max-speakers`.

## Outputs

- `data/audio/{id}_raw.wav` and `data/audio/{id}_raw_cleaned.wav`
- `data/video/{id}_source.mp4`
- `data/transcripts/{id}.json`
- `data/shorts/{id}/stories.json` and rendered Shorts per language

## Branded subtitle and outro system

Shorts open on the selected hook. Captions use bundled Montserrat ExtraBold (or system fallbacks). Styling is high-contrast white text with karaoke highlight; `VIDEO_LAYOUT_PRESET` supports `blur_bg` and `full_crop`.

### Default neutral outro

If you do not configure a custom asset, the pipeline renders the bundled HTML card at `../poster/stitch_spiritual_video_outro_card_template/code.html` (`OUTRO_HTML_PATH`) with your `BRAND_*` text injected at capture time.

### Custom outro for your channel

Supported options (only these are implemented):

| Method | Setting | Format |
|--------|---------|--------|
| Pre-rendered clip (all languages) | `BRAND_OUTRO_CUSTOM_MP4` | MP4, H.264 video + AAC audio, 1080×1920, ~8 s |
| Pre-rendered per language | `BRAND_OUTRO_CUSTOM_DIR` | Directory containing `outro_en.mp4`, `outro_es.mp4`, `outro_ru.mp4` |
| HTML template (+ optional music) | `OUTRO_HTML_PATH`, `OUTRO_MUSIC_URL`, `OUTRO_MUSIC_PATH` | HTML 9:16 card; Playwright captures frames to MP4. No `OUTRO_MUSIC_URL` → silent audio unless you place a file at `OUTRO_MUSIC_PATH`. |

Priority: `BRAND_OUTRO_CUSTOM_MP4` → `BRAND_OUTRO_CUSTOM_DIR/outro_<lang>.mp4` → HTML render to `assets/outro_rendered_<lang>.mp4`.

Custom HTML templates should keep the element ids used by `modules/outro.py`: `outro-lecture-cta`, `outro-secondary-cta` (legacy alias `outro-patreon-cta`), `outro-scholar-badge`, `outro-headline`, `outro-tagline`, `outro-brand-website`, `outro-brand-handle`, `outro-footer-copyright`.

Place optional assets under `pipeline/assets/outro/custom/` (see `.env.example`).

On HTML render, `modules/outro.py` muxes optional bed music. Set `OUTRO_MUSIC_URL` to download a track with yt-dlp, or place your own licensed file at `OUTRO_MUSIC_PATH`. With neither configured, a silent track is generated. Playwright needs `python -m playwright install chromium`.

Set `BRAND_OUTRO_TRANSITION_SECONDS` between 0.5 and 2.5 to tune the crossfade from lecture footage to outro.

```powershell
python -c "from modules.outro import ensure_outro_video; print(ensure_outro_video())"
```

Re-render saved clips without re-running clip selection:

```powershell
python apply_brand_style.py VIDEO_ID
python apply_brand_style.py
```

## VRAM

Models load one stage at a time; the pipeline drops each model and clears CUDA cache between stages. When file logging is enabled, logs are written under `logs/phase1.log`.

## Troubleshooting

- **HTTP 429:** add `cookies.txt` and keep the sleep interval.  
- **CUDA missing:** reinstall torch from the `cu128` index, or `DEVICE=cpu` / `--device cpu` for CPU Whisper (int8).  
- **Gated model error:** accept the three Hugging Face model agreements for your `HF_TOKEN`.
