#!/bin/bash
sudo sed -i 's/IDLE_TIMEOUT_SECONDS=1800/IDLE_TIMEOUT_SECONDS=300/' /etc/systemd/system/hotword-watchdog.service
sudo sed -i 's/BOOT_GRACE_SECONDS=900/BOOT_GRACE_SECONDS=300/' /etc/systemd/system/hotword-watchdog.service
sudo systemctl daemon-reload
sudo systemctl restart hotword-watchdog
echo "=== watchdog env now ==="
grep -E 'IDLE_TIMEOUT|BOOT_GRACE' /etc/systemd/system/hotword-watchdog.service
echo "=== watchdog active? ==="
systemctl is-active hotword-watchdog
