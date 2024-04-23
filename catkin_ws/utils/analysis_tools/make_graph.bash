#!/bin/bash


# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$0")

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No directory path provided."
    exit 1
fi

RESULT_PATH="$1"

# Check if the directory exists
if [ ! -d "$RESULT_PATH" ]; then
    echo "Error: Directory '$RESULT_PATH' does not exist."
    exit 1
fi

mkdir -p "$RESULT_PATH/Graph$(basename $(pwd))_png"
mkdir -p "$RESULT_PATH/Graph$(basename $(pwd))_pdf"

python3 $SCRIPT_DIR/eval_localization.py "$RESULT_PATH/Path_1"
python3 $SCRIPT_DIR/eval_localization.py "$RESULT_PATH/Path_2"
python3 $SCRIPT_DIR/eval_localization.py "$RESULT_PATH/Path_3"
python3 $SCRIPT_DIR/eval_localization.py "$RESULT_PATH/Path_4"
python3 $SCRIPT_DIR/eval_localization.py "$RESULT_PATH/Path_5"
python3 $SCRIPT_DIR/eval_localization.py "$RESULT_PATH/Path_6"
python3 $SCRIPT_DIR/eval_localization.py "$RESULT_PATH/Path_7"
python3 $SCRIPT_DIR/eval_localization.py "$RESULT_PATH/Path_8"
python3 $SCRIPT_DIR/eval_localization.py "$RESULT_PATH/Path_9"
python3 $SCRIPT_DIR/eval_localization.py "$RESULT_PATH/Path_10"
python3 $SCRIPT_DIR/eval_localization.py "$RESULT_PATH/Path_11"

mv $SCRIPT_DIR/*.png $RESULT_PATH/Graph$(basename $(pwd))_png/
mv $SCRIPT_DIR/*.pdf $RESULT_PATH/Graph$(basename $(pwd))_pdf/