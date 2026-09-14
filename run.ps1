# ============================================================
# run.ps1 — launcher for object_detection_tracking
# Run from:  PS C:\Users\shiva\OneDrive\Desktop\task 4>
# Usage:
#   .\run.ps1                          # webcam
#   .\run.ps1 --source data/sample.mp4 # video file
#   .\run.ps1 --no-db                  # skip DB logging
# ============================================================

$projectDir = Join-Path $PSScriptRoot "object_detection_tracking"
Set-Location $projectDir
python main.py @args
