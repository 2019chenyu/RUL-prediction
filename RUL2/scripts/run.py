import os
import sys
import numpy as np

print("Test 06100940所有优化器是sgd的程序都在第一个优化器loss nan了")
print("Test 06121730 调小学习率，使用sgd")

#lr_list = [0.001]
batch_size_list = [5, 30]
nb_filters_list = [32, 64, 128]
nb_stacks_list = [1,2,3,4]
dropout_rate_list = [0, 0.2, 0.5]
optimizer_list = ['sgd']
test_num = 1

for batch_size in batch_size_list:
    for nb_filters in nb_filters_list:
        for nb_stacks in nb_stacks_list:
            for dropout_rate in dropout_rate_list:
                for optimizer in optimizer_list:
                    print("Test ", test_num)
                    test_num += 1
                    print("***********************************************************************")
                    sys.stdout.flush()
                    os.system("python main.py --batch_size {} --nb_filters {} --nb_stacks {} --dropout_rate {} --optimizer {} --lr 0.0001 --epochs 100 --seed 10 --weight_decay 1e-4 --sequence_length 100 ".format(batch_size, nb_filters, nb_stacks, dropout_rate, optimizer))

