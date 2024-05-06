#!/bin/bash

pip install --upgrade build
cd package
rm -rf dist/*
python3 -m build
