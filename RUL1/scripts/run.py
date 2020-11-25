import os
import sys
import numpy as np

print("Test 0608, grid search hyperparameters")
lr_list = [0.001, 0.01]
weight_decay_list = [1e-4]
lstm1_list = [30, 60, 80]
lstm2_list = [40, 60, 80, 100]

test_num = 1
lstm1_list = [30, 30, 30, 60, 60, 60, 80, 80, 80, 80]
lstm2_list = [ 40,  60, 100,  40,  60,  80,  40,  60,  80, 100]
for lstm1, lstm2 in zip(lstm1_list, lstm2_list):
    print("Test ", test_num)
    test_num += 1
    print("***********************************************************************")
    sys.stdout.flush()
    os.system("python 1dcov+lstm+attention.py --batch_size 5 --lr 0.001 --epochs 100 --seed 10 --weight_decay 1e-4 --sequence_length 100 --lstm1 {} --lstm2 {} --dense1 50 --filters1 16 --filters2 32 --dropout1 0.6 --dropout2 0.6".format(lstm1, lstm2))

