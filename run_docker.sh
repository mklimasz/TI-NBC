#!/usr/bin/env bash

DIR=$(pwd)
docker run -v "$DIR"/data/:/data nbc:latest bash -c "nbc --path /data/s2.txt --k 10 -o /data/results.csv"