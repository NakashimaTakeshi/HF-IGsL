#!/bin/bash

# 引数から CURRENT_PATH を取得、指定されていない場合は現在のディレクトリを使用
CURRENT_PATH=${1:-$(pwd)}

if [ $# -lt 4 ]; then
    echo "引数が不足しています。"
    echo -n "データセット名と5つのエピソード番号を入力してください: "
    read input
fi

# 引数から各種変数を設定
dataset=${2:-dataset1}
mcl_epi=${3:-0}
emcl_epi=${4:-0}
model1_epi=${5:-0}
model2_epi=${6:-0}

# ファイル検索と検証を関数で定義
find_and_verify() {
    local search_pattern=$1
    local episode=$2
    local exclude_pattern=$3
    local files
    files=$(find "${CURRENT_PATH}" -maxdepth 1 -name '*.mp4' | awk -v pat="${search_pattern}" -v epi="${episode}" -v ex_pat="${exclude_pattern}" '$0 ~ pat && $0 ~ epi && ($0 !~ ex_pat || length(ex_pat) == 0)')
    # echo " ******* Found Files: "
    # echo $(find "${CURRENT_PATH}" -maxdepth 1 -name '*.mp4'| awk -v pat="17_replace.mp4")
    # echo $files
    # echo " ******************************************************************************************* Found Files: "
    files=$(echo "${files}" | tr -d '\n') # 改行を削除

    # ファイルが一つだけであることを確認
    IFS=$'\n' read -rd '' -a file_arr <<<"$files"
    if [ ${#file_arr[@]} -ne 1 ]; then
        echo "Error: File is not unique or does not exist"
        exit 1
    fi
    echo "${file_arr[0]}"
}

# 上の階層に新しいディレクトリを作成
NEW_DIR="${CURRENT_PATH}/../videos"
mkdir -p "${NEW_DIR}"
cd "${NEW_DIR}"
NEW_DIR=$(pwd)

echo "++++++++++++++ start combine ++++++++++++++"
echo "Dataset: ${dataset}"
echo "MCL Episode: ${mcl_epi}"
echo "EMCL Episode: ${emcl_epi}"
echo "Model1 Episode: ${model1_epi}"
echo "Model2 Episode: ${model2_epi}"
echo "Output Directory: ${NEW_DIR}"
echo "+++++++++++++++++++++++++++++++++++++++++++"

# 個々の動画ファイルを検索し、新しいディレクトリにコピー
MCL=$(find_and_verify "${dataset}_.*record_amcl.*" "_${mcl_epi}_" "RecoveryParamON")
echo " ******* MCL File: ${MCL}"
cp "${MCL}" "${NEW_DIR}/"
# echo "cp "${MCL}" "${NEW_DIR}/""

EMCL=$(find_and_verify "${dataset}_.*record_emcl.*" "_${emcl_epi}.mp")
echo " ******* EMCL File: ${EMCL}"
cp "${EMCL}" "${NEW_DIR}/"

MODEL1=$(find_and_verify "${dataset}_.*RSSM_node.*replace.*" "_${model1_epi}_" "MRSSM")
echo " ******* MODEL1 File: ${MODEL1}"
cp "${MODEL1}" "${NEW_DIR}/"

MODEL2=$(find_and_verify "${dataset}_.*RSSM_node_MRSSM.*replace.*" "_${model2_epi}_")
echo " ******* MODEL2 File: ${MODEL2}"
cp "${MODEL2}" "${NEW_DIR}/"

# 4 つの動画を横に並べて縦に結合する
output_file="${NEW_DIR}/${dataset}_combined.mp4"
ffmpeg -y -i "${MODEL2}" -i "${MODEL1}" -i "${MCL}" -i "${EMCL}" -filter_complex "[0:v][2:v]hstack[top];[1:v][3:v]hstack[bottom];[top][bottom]vstack" "${output_file}"

echo "Videos combined and saved to ${output_file}"
