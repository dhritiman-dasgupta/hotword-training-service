#!/bin/bash
source /opt/hotword/venvs/train/bin/activate
pip install -q "pyarrow==12.0.1"
python -c "import datasets; print('datasets', datasets.__version__, 'OK')"
