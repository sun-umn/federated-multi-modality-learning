#!/bin/bash

## install NVDlare in a virtual environment
apt update
apt-get install python3-venv

source nvflare-env/bin/activate || python -m venv nvflare-env && source nvflare-env/bin/activate 

python3 -m pip install -U pip
python3 -m pip install -U setuptools

python3 -m pip install nvflare==2.2.1
python3 -m pip install tensorboard
python3 -m pip install torch torchvision transformers
python3 -m pip install pandas
python3 -m pip install seqeval