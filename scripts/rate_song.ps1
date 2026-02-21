param(
    [Parameter(Mandatory = $true)]
    [string]$TrackPath,
    [string]$Prompt = "",
    [ValidateRange(1,5)]
    [int]$Rating = 3,
    [string]$Style = "",
    [int]$Bpm = 0,
    [string]$Notes = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python scripts\ai_audio_bridge_server.py `
  --feedback-track $TrackPath `
  --prompt $Prompt `
  --feedback-rating $Rating `
  --feedback-style $Style `
  --feedback-bpm $Bpm `
  --feedback-notes $Notes `
  --auto-learn

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host "Feedback saved and profile updated."
