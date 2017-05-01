#!/usr/bin/env bash

echo $0

#for i in 1 2 3 4 5; do
#
#    folder=../data/OSHUMEDQueryLevelNorm/Fold${i}/
#    echo $folder "started"
#    rm -rf ${folder}trainlog_ListMLE.txt
#    python ListMLE_conv2d.py --training_set ${folder}train.txt --test_set ${folder}test.txt --valid_set ${folder}vali.txt --test_output ${folder}testoutput.txt --train_output ${folder}trainoutput.txt --valid_output ${folder}validoutput.txt --model_path ${folder}model.txt --eval_output ${folder}evaloutput.txt > ${folder}trainlog_ListMLE.txt
#    python plot_ListMLE.py ${folder}trainlog_ListMLE.txt
#    echo $folder "finished"
#done

for rand in 1; do
    for iter in 1; do
        for i in 1 2 3 4 5; do
            folder=../data/OSHUMEDQueryLevelNorm/Fold${i}/
            echo  ${folder}trainlog_ListMLE.txt.${rand}.${iter} "started"
            rm -rf ${folder}trainlog_ListMLE.txt.${rand}.${iter}
            python ListMLE_conv2d.py --training_set ${folder}train.txt --test_set ${folder}test.txt --valid_set ${folder}vali.txt --test_output ${folder}testoutput.txt --train_output ${folder}trainoutput.txt --valid_output ${folder}validoutput.txt --model_path ${folder}model.txt --eval_output ${folder}evaloutput.txt --random_level ${rand} --iter ${iter} > ${folder}trainlog_ListMLE.txt.${rand}.${iter}
            python plot_ListMLE.py ${folder}trainlog_ListMLE.txt.${rand}.${iter}
            echo  ${folder}trainlog_ListMLE.txt.${rand}.${iter} "finished"
        done
    done
done