#!/usr/bin/env bash

echo $0

for i in 1 2 3 4 5; do
    folder=../data/OSHUMEDQueryLevelNorm/Fold${i}/
    echo $folder "started"
    rm -rf ${folder}trainlog_RankSVM.txt
    python RankSVM.py --training_set ${folder}train.txt --test_set ${folder}test.txt --valid_set ${folder}vali.txt --test_output ${folder}testoutput_RankSVM.txt --train_output ${folder}trainoutput_RankSVM.txt --valid_output ${folder}validoutput_RankSVM.txt --model_path ${folder}model_RankSVM.txt --eval_output ${folder}evaloutput_RankSVM.txt > ${folder}trainlog_RankSVM.txt
    python plot_ListMLE.py ${folder}trainlog_RankSVM.txt
    echo $folder "finished"
done