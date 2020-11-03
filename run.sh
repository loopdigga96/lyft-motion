#!/bin/bash

if [ -n "${wheel_dir}" ]; then
  echo "======================="
  echo "INSTALL WHEELS"
  echo "======================="

  echo "Installing wheels from '$wheel_dir'...."
  export PATH=$PATH:$HOME/.local/bin
  WHEELS=$(cd $wheel_dir; ls -1 *.whl | awk -F - '{ gsub("_", "-", $1); print $1 }' | uniq)
  pip install --user --no-index --find-links=$wheel_dir $WHEELS
fi

echo "================"
echo "MAIN"
echo "================"


python train.py