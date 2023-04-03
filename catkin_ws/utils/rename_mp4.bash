#!/usr/bin/env bash

# rename -v "/s/model1_replase/model1_likelihood/"

# SUFFIX=model1_likelihood
# SUFFIX=model1_replase
# SUFFIX=model2_likelihood
# SUFFIX=model2_replase
# SUFFIX=mcl

if [ $# -eq 0 ]; then
    echo "引数がありません。"
    echo -n "入力してください: "
    read input
else
    SUFFIX=$1
fi

echo "入力された値は $SUFFIX です。"



mv dataset1_*.mp4 Path_A_${SUFFIX}.mp4
mv dataset2_*.mp4 Path_B_${SUFFIX}.mp4
# mv dataset3_*.mp4 Path_%_${SUFFIX}.mp4
mv dataset4_*.mp4 Path_C_${SUFFIX}.mp4
mv dataset5_*.mp4 Path_D_${SUFFIX}.mp4
mv dataset6_*.mp4 Path_E_${SUFFIX}.mp4
mv dataset7_*.mp4 Path_F_${SUFFIX}.mp4
mv dataset8_*.mp4 Path_G_${SUFFIX}.mp4
mv dataset9_*.mp4 Path_H_${SUFFIX}.mp4
mv dataset10_*.mp4 Path_I_${SUFFIX}.mp4

# mkdir -p "tmp"
# mv *.mp4 "tmp"



# ffmpeg -i input1.mp4 -i input2.mp4 -i input3.mp4 -i input4.mp4 -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack" output.mp4