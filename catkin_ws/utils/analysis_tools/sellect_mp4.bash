#!/usr/bin/env bash
CURRENT_PATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# if [ $# -eq 0 ]; then
#     echo "引数がありません。"
#     echo -n "データセット名と5つのエピソード番号を入力してください: "
#     read input
# else
#     SUFFIX=$1
# fi
dataset=${1:-dataset1}
mcl_epi=${2:-0}
model1_l_epi=${3:-0}
model1_r_epi=${4:-0}
model2_l_epi=${5:-0}
model2_r_epi=${6:-0}
echo "find ${CURRENT_PATH} -maxdepth 1 -name '*.mp4' | awk '/${dataset}_/ && /_record_amcl_/ && /_${mcl_epi}_/'"
File1=$(find ${CURRENT_PATH} -maxdepth 1 -name '*.mp4' | awk '{/${dataset}_/ && /_record_amcl_/ && /_${mcl_epi}_/}')
BAGS=$(find ${BAG_DIRECTORY} -maxdepth 1 -name '*.bag' | awk '{print $0 " " substr($0,match($0,/[0-9]+/))}' | sort -k2 -n | cut -f 1 -d " ")

echo ${File1}
if [ ${#File1[@]} -ne 1 ]; then
    echo "File 1 is not unique"
    exit 1
fi

File2=$(find ${CURRENT_PATH} -maxdepth 1 -name '*.mp4' | awk '/${dataset}_/ && /RSSM_node/ && !/MRSSM/ && /likelihood/ && /_${model1_l_epi}_/')
echo ${File2}
if [ ${#File2[@]} -ne 1 ]; then
    echo "File 2 is not unique"
    exit 1
fi




# 5 つの動画を横に並べる
# do
#     ffmpeg -y -i ${File1} -i ${i}${SUFFIX_1}.mp4 -i ${i}${SUFFIX_2}.mp4 -i ${i}${SUFFIX_3}.mp4 -i ${i}${SUFFIX_4}.mp4 -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack[all];[4:v]pad=iw+200:ih*2:color=white[ref];[ref][all]hstack" ${dataset}.mp4
# done
