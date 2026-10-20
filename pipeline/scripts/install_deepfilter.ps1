# Install DeepFilterNet into the active Python environment (use pipeline\.venv when possible).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Using Python: $Python"
& $Python -m pip install -r (Join-Path $Root "requirements-deepfilter.txt") --no-deps
& $Python -m pip install "appdirs>=1.4,<2.0" "requests>=2.27,<3"
& $Python -c "from df.enhance import init_df; print('DeepFilterNet OK:', init_df)"
