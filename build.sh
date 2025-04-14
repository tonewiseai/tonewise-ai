#!/usr/bin/env bash

# Install Python if not already
apt-get update && apt-get install -y python3 python3-pip

# Upgrade pip for Python 3
python3 -m pip install --upgrade pip

# Install from requirements.txt
python3 -m pip install -r requirements.txt

