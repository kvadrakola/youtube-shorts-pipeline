[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Source,

    [Parameter()]
    [string]$Prompt,

    [Parameter()]
    [string]$PromptFile
)

$ErrorActionPreference = "Stop"

if ($PSBoundParameters.ContainsKey("Prompt") -and $PSBoundParameters.ContainsKey("PromptFile")) {
    throw "Use either -Prompt or -PromptFile, not both."
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PipelineRoot = Join-Path $Root "pipeline"
$Python = Join-Path $PipelineRoot ".venv\Scripts\python.exe"
$Script = Join-Path $PipelineRoot "tools\transcribe_and_analyze.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing $Python — create pipeline\.venv per pipeline\README.md"
}

$Arguments = @($Script, $Source)
if ($PSBoundParameters.ContainsKey("Prompt")) {
    $Arguments += @("--prompt", $Prompt)
}
elseif ($PSBoundParameters.ContainsKey("PromptFile")) {
    $Arguments += @("--prompt-file", $PromptFile)
}

& $Python @Arguments
exit $LASTEXITCODE
