#!/usr/bin/env bash
echo $0

for rand in 21; do
for i in 1 2 3 4 5; do
    for iter in 1 2; do

            folder=../data/TD2003/Fold${i}/
#            echo  ${folder}trainlog_ListMLE.txt.${rand}.${iter} "started"
            rm -rf ${folder}trainlog_ListMLE.txt.${rand}.${iter}
            python ListMLE_conv2d.py --training_set ${folder}trainingset.txt --test_set ${folder}testset.txt --valid_set ${folder}validationset.txt --test_output ${folder}testoutput.txt --train_output ${folder}trainoutput.txt --valid_output ${folder}validoutput.txt --model_path ${folder}model.txt --eval_output ${folder}evaloutput.txt --random_level ${rand} --iter ${iter} > ${folder}trainlog_ListMLE.txt.${rand}.${iter}
            python plot_ListMLE.py ${folder}trainlog_ListMLE.txt.${rand}.${iter}
#            echo  ${folder}trainlog_ListMLE.txt.${rand}.${iter} "finished"
        done
    done
done