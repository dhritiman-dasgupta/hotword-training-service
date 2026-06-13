#!/usr/bin/env python3
"""
Idle watchdog: stops the EC2 instance when it has been idle long enough and no
training job is queued/running. Because the instance is launched with
instance-initiated-shutdown-behavior=stop (and spot interruption-behavior=stop),
`shutdown -h now` STOPS the instance (preserving the EBS volume + models), it does
not terminate it. Restart later with `aws ec2 start-instances`.

Idle = (now - last_activity) > IDLE_TIMEOUT_SECONDS, where last_activity is bumped
by every API request. A boot grace period avoids shutting down right after start.
"""
import os
import time

STATE = "/opt/hotword/state"
JOBS = "/opt/hotword/jobs"
LAST_ACTIVITY = os.path.join(STATE, "last_activity")

IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "1800"))   # 30 min
BOOT_GRACE = int(os.environ.get("BOOT_GRACE_SECONDS", "900"))        # 15 min
CHECK_EVERY = 60


def boot_uptime():
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except Exception:
        return 1e9


def jobs_active():
    if not os.path.isdir(JOBS):
        return False
    import json
    for jid in os.listdir(JOBS):
        sp = os.path.join(JOBS, jid, "status.json")
        if os.path.exists(sp):
            try:
                with open(sp) as f:
                    st = json.load(f)
                if st.get("state") in ("queued", "running"):
                    return True
            except Exception:
                pass
    return False


def last_activity():
    if os.path.exists(LAST_ACTIVITY):
        try:
            return int(open(LAST_ACTIVITY).read().strip())
        except Exception:
            return 0
    return 0


def log(msg):
    print(f"[watchdog] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def main():
    os.makedirs(STATE, exist_ok=True)
    if not os.path.exists(LAST_ACTIVITY):
        with open(LAST_ACTIVITY, "w") as f:
            f.write(str(int(time.time())))
    log(f"started; idle_timeout={IDLE_TIMEOUT}s boot_grace={BOOT_GRACE}s")
    while True:
        time.sleep(CHECK_EVERY)
        if boot_uptime() < BOOT_GRACE:
            continue
        if jobs_active():
            continue
        idle = int(time.time()) - last_activity()
        if idle > IDLE_TIMEOUT:
            log(f"idle {idle}s > {IDLE_TIMEOUT}s and no active jobs -> stopping instance")
            os.system("sudo shutdown -h now")
            # give the system time to halt; loop will end with the process
            time.sleep(120)


if __name__ == "__main__":
    main()
