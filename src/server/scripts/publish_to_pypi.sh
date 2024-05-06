#!/bin/bash

pip install --upgrade twine
cd package
python3 -m twine upload dist/*
