#!/usr/bin/env bash
echo $0

for i in 1 2 3 4 5; do
    folder=../data/TD2004/Fold${i}/
    echo $folder "started"
    #rm -rf ${folder}trainlog_ListMLE.txt
    #python ListMLE_conv2d.py --training_set ${folder}trainingset.txt --test_set ${folder}testset.txt --valid_set ${folder}validationset.txt --test_output ${folder}testoutput.txt --train_output ${folder}trainoutput.txt --valid_output ${folder}validoutput.txt --model_path ${folder}model.txt --eval_output ${folder}evaloutput.txt > ${folder}trainlog_ListMLE.txt
    rm -rf ${folder}trainlog_RankSVM.txt
    python RankSVM.py --training_set ${folder}trainingset.txt --test_set ${folder}testset.txt --valid_set ${folder}validationset.txt --test_output ${folder}testoutput_RankSVM.txt --train_output ${folder}trainoutput_RankSVM.txt --valid_output ${folder}validoutput_RankSVM.txt --model_path ${folder}model_RankSVM.txt --eval_output ${folder}evaloutput_RankSVM.txt > ${folder}trainlog_RankSVM.txt
    python plot_ListMLE.py ${folder}trainlog_RankSVM.txt
    echo $folder "finished"
done