# Run Phase 1 with pipeline\.venv (Python 3.11 + CUDA stack from README).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing $Python — create .venv per pipeline/README.md"
}
& $Python (Join-Path $Root "run_ingestion_pipeline.py") @args
exit $LASTEXITCODE
