

mkdir -p "Graph$(basename $(pwd))_png"
mkdir -p "Graph$(basename $(pwd))_pdf"

python3 eval_localization.py "Path A"
python3 eval_localization.py "Path B"
python3 eval_localization.py "Path C"
python3 eval_localization.py "Path D"
python3 eval_localization.py "Path E"
python3 eval_localization.py "Path F"
python3 eval_localization.py "Path G"
python3 eval_localization.py "Path H"
python3 eval_localization.py "Path I"

mv *.png Graph$(basename $(pwd))_png/
mv *.pdf Graph$(basename $(pwd))_pdf/