folder=../data/MQ2008/Fold1/
python ListMLE_conv2d.py --training-set ${folder}train.txt --test-set ${folder}test.txt --validation-set ${folder}valid.txt --test_output ${folder}testoutput.txt --train_output ${folder}trainoutput.txt --model_path ${folder}model.txt --eval_output ${folder}evaloutput.txt
