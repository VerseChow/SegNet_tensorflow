#!/bin/bash
set -x
set -e

if [ ! -d ./Data/train ]
	then
	echo "[INFO] Dataset for training does not exist. Trying to convert from VOC2012..."
	if [ ! -d ./VOCdevkit/VOC2012 ]
		then
		echo "[INFO] VOC2012 not found. Exiting..."
		exit 1
	else
		python ./scripts/VOC2012_devkit.py
	fi
fi

echo "[INFO] Start Training..."
python ./scripts/main.py

set +x