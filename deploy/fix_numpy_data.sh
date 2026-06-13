#!/bin/bash
set -e
source /opt/hotword/venvs/train/bin/activate
pip install -q "numpy<2" "pyarrow==12.0.1"
python -c "import numpy, pyarrow, datasets; print('numpy', numpy.__version__, '| pyarrow', pyarrow.__version__, '| datasets', datasets.__version__, 'OK')"
python -c "import openwakeword.data; print('oww.data still OK')"
echo "RUN_DATA_PREP"
nohup /opt/hotword/venvs/train/bin/python /tmp/dl_data.py > /opt/hotword/dl_data.log 2>&1 &
echo "dl_data relaunched pid=$!"
