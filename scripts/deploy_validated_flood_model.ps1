param(
  [Parameter(Mandatory=$true)][int]$TrainingPid,
  [Parameter(Mandatory=$true)][int]$ApiPid,
  [Parameter(Mandatory=$true)][string]$ArtifactRoot
)

$ErrorActionPreference = 'Stop'
$python = 'C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python39_64\python.exe'
$project = Split-Path -Parent $PSScriptRoot
$promotion = Join-Path $ArtifactRoot 'promotion.json'
$statusPath = Join-Path $ArtifactRoot 'deployment_status.json'

Wait-Process -Id $TrainingPid

# A separate evaluation watcher writes promotion.json after official test and
# Bolivia hold-out measurement. Give that independent evaluation time to finish.
$deadline = (Get-Date).AddHours(4)
while (-not (Test-Path $promotion)) {
  if ((Get-Date) -gt $deadline) { throw 'Timed out waiting for flood evaluation promotion record.' }
  Start-Sleep -Seconds 30
}

$gate = Get-Content -Raw $promotion | ConvertFrom-Json
if ($gate.status -ne 'eligible_for_manual_backend_integration') {
  @{ status = 'not_deployed'; reason = 'evaluation_gate_not_passed'; promotion = $gate } | ConvertTo-Json -Depth 6 | Set-Content $statusPath
  exit 0
}

$checkpoint = Join-Path $ArtifactRoot 'best.pt'
$onnx = Join-Path $ArtifactRoot 'model.onnx'
$env:PYTHONPATH = "$project\.runtime;C:\Users\ADMIN\OneDrive\Documents;$project"
& $python -m ml.export_flood_onnx --checkpoint $checkpoint --output $onnx --size 512
if ($LASTEXITCODE -ne 0) { throw 'ONNX export failed.' }

Stop-Process -Id $ApiPid -Force -ErrorAction SilentlyContinue
$command = "`$env:PYTHONPATH='$project\.runtime;C:\Users\ADMIN\OneDrive\Documents;$project'; `$env:RSAI_FLOOD_MODEL_ONNX='$onnx'; `$env:RSAI_FLOOD_MODEL_PROMOTION='$promotion'; Set-Location '$project'; & '$python' -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList '-NoProfile','-WindowStyle','Hidden','-Command',$command -WindowStyle Hidden
@{ status = 'deployed'; runtime = 'onnxruntime'; checkpoint = $checkpoint; onnx = $onnx; promotion = $promotion } | ConvertTo-Json -Depth 4 | Set-Content $statusPath
