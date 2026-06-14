#!/bin/bash
set -e
echo "[web] installing caddy"
sudo apt-get update -y -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q debian-keyring debian-archive-keyring apt-transport-https curl >/dev/null
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
sudo apt-get update -y -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q caddy >/dev/null
echo "[web] deploying site"
sudo rm -rf /var/www/hotword && sudo mkdir -p /var/www/hotword
sudo cp -r /tmp/web/* /var/www/hotword/
sudo chown -R caddy:caddy /var/www/hotword
echo '65-2-7-128.sslip.io {
    root * /var/www/hotword
    file_server
    encode gzip
}' | sudo tee /etc/caddy/Caddyfile >/dev/null
sudo systemctl restart caddy
sleep 4
echo "[web] caddy active:"; systemctl is-active caddy
echo "[web] files:"; ls /var/www/hotword
