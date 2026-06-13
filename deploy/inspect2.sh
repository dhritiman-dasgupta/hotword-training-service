#!/bin/bash
echo "=== train.py argparse + main (595-720) ==="
sed -n '595,720p' /opt/hotword/openWakeWord/openwakeword/train.py
echo
echo "=== all config[...] keys used in train.py ==="
grep -noE "config\[[\"'][a-zA-Z0-9_]+[\"']\]" /opt/hotword/openWakeWord/openwakeword/train.py | sort -u
echo
echo "=== config.get keys ==="
grep -noE "config.get\([\"'][a-zA-Z0-9_]+" /opt/hotword/openWakeWord/openwakeword/train.py | sort -u
echo
echo "=== sample training yaml in repo ==="
find /opt/hotword/openWakeWord -name "*.yaml" -o -name "*.yml" | xargs -I{} sh -c 'echo "--- {} ---"; cat {}' 2>/dev/null | head -120
echo
echo "=== sizes of big downloads (HEAD) ==="
for u in \
  "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" \
  "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy" \
  "https://f002.backblazeb2.com/file/openwakeword-resources/data/fma_sample.zip" \
  "https://f002.backblazeb2.com/file/openwakeword-resources/data/fsd50k_sample.zip" ; do
  sz=$(curl -sIL "$u" | grep -i content-length | tail -1)
  echo "$sz  <-  $u"
done
