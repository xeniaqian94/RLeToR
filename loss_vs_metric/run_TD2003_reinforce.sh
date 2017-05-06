#!/usr/bin/env bash
echo $0

for i in 2 3 4 5; do
    for load_model in {1..10}; do
#     for iter in 1; do
            folder=../data/TD2003/Fold${i}/
            rm -rf ${folder}trainlog_reinforce.txt.${load_model}
            python reinforce.py --training_set ${folder}trainingset.txt --test_set ${folder}testset.txt --valid_set ${folder}validationset.txt --test_output ${folder}testoutput.txt --train_output ${folder}trainoutput.txt --valid_output ${folder}validoutput.txt --load_model_path ${folder}model.txt.1.${load_model} --model_path ${folder}model.txt.reinforce.${load_model} --eval_output ${folder}evaloutput.txt.reinforce.${load_model} > ${folder}trainlog_reinforce.txt.${load_model}
            python plot_ListMLE.py ${folder}trainlog_reinforce.txt.${load_model}
        done
    done