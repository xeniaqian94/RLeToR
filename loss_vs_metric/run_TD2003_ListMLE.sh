#!/usr/bin/env bash
echo $0

for rand in 1; do
for i in 1 2 3 4 5; do
    for iter in {1..10}; do
#     for iter in 1 2; do
            folder=../data/TD2003/Fold${i}/
            rm -rf ${folder}trainlog_ListMLE.txt.${rand}.${iter}.after
            python ListMLE_conv2d.py --training_set ${folder}trainingset.txt --test_set ${folder}testset.txt --valid_set ${folder}validationset.txt --test_output ${folder}testoutput.txt --train_output ${folder}trainoutput.txt --valid_output ${folder}validoutput.txt --load_model_path ${folder}model.txt.${rand}.${iter} --model_path ${folder}model.txt.after --eval_output ${folder}evaloutput.txt --random_level ${rand} --iter ${iter} > ${folder}trainlog_ListMLE.txt.${rand}.${iter}.after
            python plot_ListMLE.py ${folder}trainlog_ListMLE.txt.${rand}.${iter}.after
        done
    done
done