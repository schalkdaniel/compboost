#!/bin/bash

filename=$1
while true
do
  free -k | awk -v date="$(date +%Y-%m-%d-%H:%M)" 'FNR == 2 {print date ": " $2 " kb"}' >> $filename
  sleep 10
done

