#!/bin/bash
# Usage: jobwait.sh <job_id>  — wait until the job finishes, print final status + model path.
JID="$1"
J="/opt/hotword/jobs/$JID"
for i in $(seq 1 180); do
  state=$(/opt/hotword/venvs/api/bin/python -c "import json;print(json.load(open('$J/status.json'))['state'])" 2>/dev/null)
  stage=$(/opt/hotword/venvs/api/bin/python -c "import json;print(json.load(open('$J/status.json')).get('stage'))" 2>/dev/null)
  echo "$(date +%H:%M:%S) state=$state stage=$stage"
  if [ "$state" = "done" ] || [ "$state" = "error" ]; then break; fi
  sleep 20
done
echo "=== final status ==="
cat "$J/status.json"
echo
echo "=== onnx files ==="
ls -la "$J"/*.onnx 2>/dev/null || echo "none in job dir"
ls -la "$J"/work/*.onnx 2>/dev/null || echo "none in work dir"
