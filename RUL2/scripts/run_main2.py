import os
import sys
#import numpy as np

print("Test 0609, tcn model, search tcn_num\batch_size")
lr_list = [0.0001]
batch_size_list = [5, 10, 15]
tcn_num_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
test_num = 1

for tcn_num in tcn_num_list:
    for batch_size in batch_size_list:
        for lr in lr_list:
            print("Test ", test_num)
            test_num += 1
            print("***********************************************************************")
            sys.stdout.flush()
            os.system("python main2.py --tcn_num {} --batch_size {} --lr {} --epochs 100 --seed 10 --weight_decay 1e-4 --sequence_length 100 ".format(tcn_num, batch_size, lr))

