param(
  [Parameter(Mandatory=$true)][int]$TrainingPid,
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [Parameter(Mandatory=$true)][string]$ArtifactRoot
)

$ErrorActionPreference = 'Stop'
$python = 'C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python39_64\python.exe'
$project = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$project\.runtime;$project"

Wait-Process -Id $TrainingPid
$checkpoint = Join-Path $ArtifactRoot 'best.pt'
if (-not (Test-Path $checkpoint)) { throw "Training finished without a checkpoint: $checkpoint" }

& $python -m ml.eval_flood --data $DataRoot --checkpoint $checkpoint --output (Join-Path $ArtifactRoot 'evaluation.json') --batch-size 1 --mc-passes 12
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m ml.finalize_flood --evaluation (Join-Path $ArtifactRoot 'evaluation.json') --output (Join-Path $ArtifactRoot 'promotion.json')
exit $LASTEXITCODE
