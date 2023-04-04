#!/usr/bin/env bash

# PATH SETUP
CURRENT_PATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
RESULT_PATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )'/catkin_ws/result/eval/'
MODEL_PATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )'/catkin_ws/src/ros_rssm/scripts/'
VALDATA_PATH='dataset'
ANALYSIS_FILEPATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )'/catkin_ws/utils/'

# Delete the result file
rm -rf ${RESULT_PATH}npy
rm -rf ${RESULT_PATH}movie/tmp
# $(find ${RESULT_PATH}npy/ -maxdepth 1 -name "*.npy" -type f -delete)
# $(find ${RESULT_PATH}movie/tmp/ -maxdepth 1 -name "*.mp4" -type f -delete)

for RSSM_MODEL in record_amcl RSSM_node_MRSSM RSSM_node ;do
    for INTEGRATION_MODE in replace likelihood ;do
        # Set the experiment parameters
        if [ ${RSSM_MODEL} == 'record_amcl' ]; then
            # sed -i "s/resp.integration_mode = .../resp.integration_mode = 0.0/g" ${MODEL_PATH}/RSSM_node_MRSSM.py
            INTEGRATION_MODE=none
        else
            if [ ${INTEGRATION_MODE} == 'likelihood' ]; then
                # echo "INTEGRATION_MODE = 1.0"
                sed -i "s/resp.integration_mode = .../resp.integration_mode = 1.0/g" ${MODEL_PATH}/${RSSM_MODEL}.py
                # sed -i "s/resp.integration_mode = 2.0/resp.integration_mode = ${INTEGRATION_MODE}/g" ${MODEL_PATH}/${MODEL}
            elif [ ${INTEGRATION_MODE} == 'replace' ]; then
                # echo "INTEGRATION_MODE = 2.0"
                sed -i "s/resp.integration_mode = .../resp.integration_mode = 2.0/g" ${MODEL_PATH}/${RSSM_MODEL}.py
            else
                echo "INTEGRATION_MODE = likelihood or replace"
                exit 1
            fi
        fi

        # Execute the experiment
        cd ${CURRENT_PATH}
        bash start_experiment.bash ${VALDATA_PATH} ${RSSM_MODEL}.py

        # Rename and analyze the result data
        NPY_PATH=${RESULT_PATH}/npy/${RSSM_MODEL}_${INTEGRATION_MODE}/
        mkdir -p ${NPY_PATH}
        mv ${RESULT_PATH}npy/*.npy ${NPY_PATH}

        cp ${ANALYSIS_FILEPATH}/analyse_result/* ${NPY_PATH}
        cd ${NPY_PATH}
        bash prepare_eval.bash
        bash make_graph.bash
        # # rm ${RESULT_PATH}/result.npy


        MOVIE_PATH=${RESULT_PATH}/movie/tmp/
        mkdir -p ${MOVIE_PATH}
        cd ${RESULT_PATH}/movie/
        ls *.mp4| sed "p;s/[.]mp4/_${INTEGRATION_MODE}.mp4/"|xargs -n2 mv
        mv ${RESULT_PATH}movie/*mp4 ${MOVIE_PATH}   
        # cp ${ANALYSIS_FILEPATH}/join_mp4.bash ${MOVIE_PATH}
        # cp ${ANALYSIS_FILEPATH}/rename_mp4.bash ${MOVIE_PATH}
        # cd ${MOVIE_PATH}
        # bash rename_mp4.bash ${RSSM_MODEL}_${INTEGRATION_MODE}

        #if RSSM_MODEL is MCL, break the loop        
        if [ ${RSSM_MODEL} == 'record_amcl' ]; then
            break
        fi
    done
done
