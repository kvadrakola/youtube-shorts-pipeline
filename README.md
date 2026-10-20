# Lecture-to-Shorts pipeline

Python pipeline that turns long-form YouTube (or local) lecture video into captioned 9:16 Shorts with optional branded outros and multilingual YouTube metadata.

## What it does

1. Download source video (yt-dlp) and extract 16 kHz mono audio  
2. Denoise (DeepFilterNet), transcribe (WhisperX), align words, diarize speakers (Pyannote)  
3. Select clip candidates with an LLM (DeepSeek / Gemini)  
4. Render Shorts (ffmpeg), ASS karaoke captions, optional HTML outro (Playwright)  
5. Localize metadata (ES/RU) and export description files  

See [pipeline/README.md](pipeline/README.md) for setup details.

## Tested environment (unit tests)

Verified on the sanitized public-prep copy (CPU unit tests only):

| Item | Version / note |
|------|----------------|
| OS | Windows 11 |
| Python | 3.12 |
| Tests | `pip install -r requirements-test.txt` then `python -m unittest discover -s tests` (CPU, no GPU ingest) |

Full ingest (CUDA, Whisper large-v3, Pyannote, Playwright outro render) is **not** re-run as part of publication prep. Install PyTorch with CUDA and download models on first use.

## Supported inputs (as designed)

- Public YouTube URLs via yt-dlp (cookies optional for rate limits)  
- Long-form spoken content; main speaker bias in clip selection  
- Output languages: `en`, `es`, `ru` for metadata/outro copy  

## Not guaranteed

- Every YouTube channel, age-restricted video, or music-heavy content  
- Perfect diarization with many overlapping speakers  
- All world languages (only configured `TARGET_LANGUAGES` / outro locales)  
- YouTube Terms of Service compliance for your use case — review before automating uploads  

## Quick start

```powershell
cd pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
copy .env.example .env
# Fill HF_TOKEN, DEEPSEEK_API_KEY, GEMINI_API_KEY, and BRAND_* values
python -m playwright install chromium
python run_ingestion_pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Brand configuration

| Variable | Purpose |
|----------|---------|
| `BRAND_NAME` | Channel name in SEO and tags |
| `BRAND_WEBSITE_URL` | Link in descriptions and outro |
| `BRAND_SOCIAL_HANDLE` | Outro social strip |
| `BRAND_OUTRO_*` | Outro headline, tagline, scholar badge text |
| `BRAND_OUTRO_CUSTOM_MP4` | Your own outro MP4 (skips HTML render) |
| `BRAND_OUTRO_CUSTOM_DIR` | Per-language `outro_<lang>.mp4` files |
| `OUTRO_HTML_PATH` | Custom HTML outro template (Playwright) |

Defaults use `example.com` and `Example Channel`.

## Examples

Synthetic samples (no production media): [examples/sample/](examples/sample/).

## Demo media

No demo video is bundled in the repository. Record your own neutral screen capture after configuring `BRAND_*`, or use the sample JSON/ASS files above.

## License

Application **source code** is released under the MIT License ([LICENSE](LICENSE), copyright Konstantin Mlechka). Bundled fonts and other third-party assets have separate terms — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
