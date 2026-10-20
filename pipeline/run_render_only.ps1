# Re-render saved Shorts (ASS + ffmpeg only).
# Examples:
#   .\run_render_only.ps1 "4If_qM2qUKI"
#   .\run_render_only.ps1 "4If_qM2qUKI" -Lang "es,ru"
#   .\run_render_only.ps1 --lang "en"
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing $Python — create .venv per pipeline/README.md"
}
$ForwardedArgs = @(
    foreach ($Argument in $args) {
        if ($Argument -ceq "-Lang") { "--lang" } else { $Argument }
    }
)
& $Python (Join-Path $Root "run_render_only.py") @ForwardedArgs
exit $LASTEXITCODE
