#!/bin/sh
set -e

exec python /usr/local/lib/python3.7/site-packages/mhcnuggets/src/predict.py "$@"