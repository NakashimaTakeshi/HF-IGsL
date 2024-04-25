#!/bin/bash
CURRENT_PATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ $# -eq 0 ]; then
    echo "引数がありません。"
    echo -n "SUFFIXを入力してください: "
    read input
else
    SUFFIX=${1:-RecoveryParamON}
fi



for file in *.mp4; do
  mv "$file" "${file%.mp4}_${SUFFIX}.mp4"
done