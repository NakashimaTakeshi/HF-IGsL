#!/usr/bin/env bash
SUFFIX_0=model1_likelihood
SUFFIX_1=model1_replase
SUFFIX_2=model2_likelihood
SUFFIX_3=model2_replase
SUFFIX_4=mcl

# PREFIX=Path_A_
# ffmpeg -i ${PREFIX}${SUFFIX_1}.mp4 -i ${PREFIX}${SUFFIX_2}.mp4 -i ${PREFIX}${SUFFIX_3}.mp4 -i ${PREFIX}${SUFFIX_4}.mp4 -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack" ${PREFIX}.mp4

# 5 つの動画を横に並べる
for i in Path_A_ Path_B_ Path_C_ Path_D_ Path_E_ Path_F_ Path_G_ Path_H_ Path_I_
do
    # ffmpeg -i ${i}${SUFFIX_0}.mp4 -i ${i}${SUFFIX_1}.mp4 -i ${i}${SUFFIX_2}.mp4 -i ${i}${SUFFIX_3}.mp4 -i ${i}${SUFFIX_4}.mp4 -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack[all];[4:v]pad=iw+200:ih*2:color=white[ref];[ref][all]hstack" ${i}.mp4
    ffmpeg -y -i ${i}${SUFFIX_0}.mp4 -i ${i}${SUFFIX_1}.mp4 -i ${i}${SUFFIX_2}.mp4 -i ${i}${SUFFIX_3}.mp4 -i ${i}${SUFFIX_4}.mp4 -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack[all];[4:v]pad=iw+200:ih*2:color=white[ref];[ref][all]hstack" ${i}.mp4
done

# for i in Path_A_ Path_B_ Path_C_ Path_D_ Path_E_ Path_F_ Path_G_ Path_H_ Path_I_
# do
#     ffmpeg -i ${i}${SUFFIX_1}.mp4 -i ${i}${SUFFIX_2}.mp4 -i ${i}${SUFFIX_3}.mp4 -i ${i}${SUFFIX_4}.mp4 -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack" ${i}.mp4
# done

# 2 つの動画を縦に並べる
# Filename_left=validation_ep0_predict_pose_2023_0330_142518.mp4
# Filename_right=validation_ep0_predict_pose_2023_0330_150155.mp4

# ffmpeg -y -i ${Filename_left} -i ${Filename_right} -filter_complex "[0:v][1:v]vstack" out.mp4
