#!/bin/bash
set -e
mkdir -p /opt/hotword/server /opt/hotword/state /opt/hotword/jobs
cp /tmp/srv/app.py /tmp/srv/pipeline.py /tmp/srv/f5_generate.py /tmp/srv/watchdog.py /opt/hotword/server/

# API key
if [ ! -f /opt/hotword/state/api_key ]; then
  python3 -c "import secrets;print(secrets.token_hex(24))" > /opt/hotword/state/api_key
fi
echo "API_KEY=$(cat /opt/hotword/state/api_key)"

# systemd units
sudo cp /tmp/srv/hotword-api.service /etc/systemd/system/hotword-api.service
sudo cp /tmp/srv/hotword-watchdog.service /etc/systemd/system/hotword-watchdog.service
sudo systemctl daemon-reload
sudo systemctl enable hotword-api hotword-watchdog
sudo systemctl restart hotword-api hotword-watchdog
sleep 4
echo "=== api status ==="
systemctl is-active hotword-api
echo "=== watchdog status ==="
systemctl is-active hotword-watchdog
echo "=== local health ==="
curl -s http://127.0.0.1:8000/health
echo
