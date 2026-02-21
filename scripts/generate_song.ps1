param(
    [string]$OutputPath,
    [int]$Seconds = 120,
    [ValidateSet("gemini","musicgen","local")]
    [string]$Backend = "musicgen",
    [ValidateSet("auto","synth","musicgen")]
    [string]$Engine = "musicgen",
    [string]$Model = "facebook/musicgen-small"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($Backend -eq "gemini" -and (-not $env:GEMINI_API_KEY -or [string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY))) {
    Write-Error "GEMINI_API_KEY is not set. Set it first, then run this script again."
    exit 1
}

$prompt = Read-Host "Enter your song prompt"
if ([string]::IsNullOrWhiteSpace($prompt)) {
    Write-Error "Prompt is required."
    exit 1
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputPath = "scripts\generated_song_$stamp.wav"
}

Write-Host "Generating song..."
python scripts\ai_audio_bridge_server.py `
    --generate-output $OutputPath `
    --prompt $prompt `
    --backend $Backend `
    --engine $Engine `
    --musicgen-model $Model `
    --generate-seconds $Seconds `
    --auto-learn

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Done: $OutputPath"
