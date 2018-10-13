#!/bin/bash

# Fix the symbolic link caused by building and using images on different devices.
# 
# Note: if you observe the following message when running `nvidia-smi`
# ```
# NVIDIA-SMI couldn't find libnvidia-ml.so library in your system. Please make sure that the NVIDIA Display Driver is properly installed and present in your system.
# Please also try adding directory that contains libnvidia-ml.so to your system PATH.
# ```
rm $(ldconfig 2>&1 | grep 'is empty, not checked' | awk '{print $3}') 2> /dev/null || true
