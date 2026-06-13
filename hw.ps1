<#
  hw.ps1 — control the hotword training service on AWS.

  Usage:
    .\hw.ps1 start                         # start the (stopped) instance, wait for API
    .\hw.ps1 stop                          # stop the instance (saves cost)
    .\hw.ps1 status                        # instance + API status
    .\hw.ps1 train  -Word "hey kiki"       # queue a training job  (-Samples, -Steps optional)
    .\hw.ps1 jobs                          # list jobs
    .\hw.ps1 watch  -Job <job_id>          # poll a job until it finishes
    .\hw.ps1 download -Job <job_id>        # download the trained .onnx here
    .\hw.ps1 logs   -Job <job_id>          # print full training log
    .\hw.ps1 ssh                           # open an SSH shell on the instance
#>
param(
  [Parameter(Position=0)][string]$Command = "status",
  [string]$Word,
  [string]$Job,
  [int]$Samples = 5000,
  [int]$Steps = 20000,
  [string]$Out
)

$ErrorActionPreference = "Stop"
$REGION   = "ap-south-1"
$INSTANCE = "i-089730e889eb1fb3d"
$EIP      = "13.200.68.170"
$PEM      = "$PSScriptRoot\deploy\hotword-key.pem"
$BASE     = "http://${EIP}:8000"
$KEYFILE  = "$PSScriptRoot\deploy\api_key.txt"
$APIKEY   = if (Test-Path $KEYFILE) { (Get-Content $KEYFILE -Raw).Trim() } else { "" }
$Headers  = @{}
if ($APIKEY) { $Headers["X-API-Key"] = $APIKEY }

function Instance-State {
  aws ec2 describe-instances --instance-ids $INSTANCE --region $REGION `
    --query "Reservations[0].Instances[0].State.Name" --output text
}

function Wait-Api {
  Write-Host "waiting for API at $BASE ..." -NoNewline
  for ($i=0; $i -lt 60; $i++) {
    try {
      $h = Invoke-RestMethod -Uri "$BASE/health" -TimeoutSec 5
      Write-Host " up. gpu=$($h.gpu); data_ready=$($h.data_ready)"
      return $true
    } catch { Write-Host "." -NoNewline; Start-Sleep 5 }
  }
  Write-Host " timed out."; return $false
}

switch ($Command.ToLower()) {

  "start" {
    $st = Instance-State
    Write-Host "instance state: $st"
    if ($st -ne "running") {
      aws ec2 start-instances --instance-ids $INSTANCE --region $REGION | Out-Null
      Write-Host "starting... (spot capacity permitting)"
      aws ec2 wait instance-running --instance-ids $INSTANCE --region $REGION
      Write-Host "instance running."
    }
    Wait-Api | Out-Null
  }

  "stop" {
    aws ec2 stop-instances --instance-ids $INSTANCE --region $REGION `
      --query "StoppingInstances[0].[InstanceId,CurrentState.Name]" --output text
  }

  "status" {
    Write-Host "instance: $(Instance-State)"
    try {
      $h = Invoke-RestMethod -Uri "$BASE/health" -TimeoutSec 5
      $h | ConvertTo-Json -Depth 5
    } catch { Write-Host "API not reachable (instance stopped? run: .\hw.ps1 start)" }
  }

  "train" {
    if (-not $Word) { throw "use -Word ""your wake word""" }
    $body = @{ wake_word = $Word; n_samples = $Samples; steps = $Steps } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "$BASE/train" -Method Post -Body $body `
          -ContentType "application/json" -Headers $Headers
    Write-Host "queued job: $($r.job_id)"
    Write-Host "watch:    .\hw.ps1 watch -Job $($r.job_id)"
    Write-Host "download: .\hw.ps1 download -Job $($r.job_id)"
  }

  "jobs" {
    Invoke-RestMethod -Uri "$BASE/jobs" -Headers $Headers | Format-Table -AutoSize
  }

  "watch" {
    if (-not $Job) { throw "use -Job <job_id>" }
    while ($true) {
      $s = Invoke-RestMethod -Uri "$BASE/jobs/$Job" -Headers $Headers
      Write-Host ("[{0}] state={1} stage={2} progress={3}%" -f (Get-Date -Format HH:mm:ss), $s.state, $s.stage, $s.progress)
      if ($s.state -eq "done") { Write-Host "DONE. download: .\hw.ps1 download -Job $Job"; break }
      if ($s.state -eq "error") { Write-Host "ERROR: $($s.error)"; Write-Host $s.log_tail; break }
      Start-Sleep 15
    }
  }

  "download" {
    if (-not $Job) { throw "use -Job <job_id>" }
    if (-not $Out) { $Out = "$PSScriptRoot\models\$Job.onnx" }
    New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
    Invoke-WebRequest -Uri "$BASE/jobs/$Job/model" -Headers $Headers -OutFile $Out
    Write-Host "saved model -> $Out"
  }

  "logs" {
    if (-not $Job) { throw "use -Job <job_id>" }
    Invoke-RestMethod -Uri "$BASE/jobs/$Job/log" -Headers $Headers
  }

  "ssh" {
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -i $PEM "ubuntu@$EIP"
  }

  default { Write-Host "unknown command '$Command'. See the header of hw.ps1 for usage." }
}
