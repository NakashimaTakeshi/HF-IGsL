#!/bin/bash
CURRENT_PATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ $# -eq 0 ]; then
    echo "引数がありません。"
    echo -n "データセット名と5つのエピソード番号を入力してください: "
    read input
else
    SUFFIX=$1
fi
dataset=${1:-dataset1}
mcl_epi=${2:-0}
mcl_epi2=${3:-0}
model1_l_epi=${4:-0}
model1_r_epi=${5:-0}
model2_l_epi=${6:-0}
model2_r_epi=${7:-0}

# File0=$(find ${CURRENT_PATH} -maxdepth 1 -name '*.mp4' | awk '/'"${dataset}"'_/ && /record_amcl/ && !/RecoveryParamON/ && /_'"${mcl_epi}"'_/')
File0=$(find ${CURRENT_PATH} -maxdepth 1 -name '*.mp4' | awk '/'"${dataset}"'_/ && /record_amcl/ && /_'"${model1_l_epi}"'_/')
echo ${File0}
file_arr=(${File0})
# echo ${#file_arr[@]}
if [ ${#file_arr[@]} -ne 1 ]; then
    echo "File 0 is not unique"
    exit 1
fi

File1=$(find ${CURRENT_PATH} -maxdepth 1 -name '*.mp4' | awk '/'"${dataset}"'_/ && /RSSM_node/ && !/MRSSM/ && /likelihood/ && /_'"${model1_l_epi}"'_/')
echo ${File1}
file_arr=(${File1})
# echo ${#file_arr[@]}
if [ ${#file_arr[@]} -ne 1 ]; then
    echo "File 1 is not unique"
    exit 1
fi

File2=$(find ${CURRENT_PATH} -maxdepth 1 -name '*.mp4' | awk '/'"${dataset}"'_/ && /RSSM_node_MRSSM/ && /likelihood/ && /_'"${model2_l_epi}"'_/')
echo ${File2}
file_arr=(${File2})
# echo ${#file_arr[@]}
if [ ${#file_arr[@]} -ne 1 ]; then
    echo "File 2 is not unique"
    exit 1
fi

File3=$(find ${CURRENT_PATH} -maxdepth 1 -name '*.mp4' | awk '/'"${dataset}"'_/ && /RSSM_node/ && !/MRSSM/ && /replace/ && /_'"${model1_r_epi}"'_/')
echo ${File3}
file_arr=(${File3})
# echo ${#file_arr[@]}
if [ ${#file_arr[@]} -ne 1 ]; then
    echo "File 3 is not unique"
    exit 1
fi

File4=$(find ${CURRENT_PATH} -maxdepth 1 -name '*.mp4' | awk '/'"${dataset}"'_/ && /RSSM_node_MRSSM/ && /replace/ && /_'"${model2_r_epi}"'_/')
echo ${File4}
file_arr=(${File4})
# echo ${#file_arr[@]}
if [ ${#file_arr[@]} -ne 1 ]; then
    echo "File 4 is not unique"
    exit 1
fi

File5=$(find ${CURRENT_PATH} -maxdepth 1 -name '*.mp4' | awk '/'"${dataset}"'_/ && /record_amcl/ && /_RecoveryParamON/ && /_'"${mcl_epi2}"'_/')
echo ${File5}
file_arr=(${File5})
# echo ${#file_arr[@]}
if [ ${#file_arr[@]} -ne 1 ]; then
    echo "File 5 is not unique"
    exit 1
fi



# # 5 つの動画を横に並べる
echo "ffmpeg -y -i ${File1} -i ${File2} -i ${File3} -i ${File4} -i ${File0} -filter_complex [0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack[all];[4:v]pad=iw+200:ih*2:color=white[ref];[ref][all]hstack ${dataset}.mp4"
ffmpeg -y -i ${File1} -i ${File2} -i ${File3} -i ${File4} -i ${File0} -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack[all];[4:v]pad=iw+200:ih*2:color=white[ref];[ref][all]hstack" ${dataset}.mp4

# 6 つの動画を横に並べる
# ffmpeg -y -i ${File1} -i ${File2} -i ${File3} -i ${File4} -i ${File0} -i ${File5} -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack[all];[4:v][5:v]vstack[ref1];[ref1]pad=iw+200:ih:color=white[ref];[ref][all]hstack" ../${dataset}.mp4
