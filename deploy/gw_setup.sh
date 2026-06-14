#!/bin/bash
set -e
echo "[gw] python venv + deps"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.10-venv python3-pip >/dev/null
sudo mkdir -p /opt/gateway/cache
sudo cp /tmp/gateway.py /opt/gateway/gateway.py
sudo cp /tmp/gw_api_key /opt/gateway/api_key
sudo cp /tmp/gw_api_key /opt/gateway/gpu_api_key
sudo chown -R ubuntu:ubuntu /opt/gateway
[ -d /opt/gateway/venv ] || python3 -m venv /opt/gateway/venv
/opt/gateway/venv/bin/pip install -q --upgrade pip
/opt/gateway/venv/bin/pip install -q fastapi "uvicorn[standard]" boto3 requests python-multipart

echo "[gw] systemd unit"
sudo tee /etc/systemd/system/hotword-gateway.service >/dev/null <<'UNIT'
[Unit]
Description=Hotword Gateway API
After=network-online.target
Wants=network-online.target
[Service]
User=ubuntu
WorkingDirectory=/opt/gateway
ExecStart=/opt/gateway/venv/bin/uvicorn gateway:app --host 127.0.0.1 --port 8001
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable hotword-gateway >/dev/null 2>&1
sudo systemctl restart hotword-gateway

echo "[gw] caddy: site + /api proxy"
sudo tee /etc/caddy/Caddyfile >/dev/null <<'CADDY'
65-2-7-128.sslip.io {
    handle_path /api/* {
        reverse_proxy 127.0.0.1:8001
    }
    handle {
        root * /var/www/hotword
        file_server
        encode gzip
    }
}
CADDY
sudo systemctl reload caddy || sudo systemctl restart caddy
sleep 4
echo "[gw] gateway active:"; systemctl is-active hotword-gateway
echo "[gw] caddy active:"; systemctl is-active caddy
echo "[gw] local health:"; curl -s http://127.0.0.1:8001/health; echo
