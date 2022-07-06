import pickle
import copy
import itertools
import numpy as np
from natsort import natsorted

with open('3dmm_data/param_all_norm_v201.pkl', 'rb') as p:
    param_all_norm_v201 = pickle.load(p)

with open('3dmm_data/train_aug_120x120.list.train') as f:
    train_list = f.read().splitlines()

param_all_norm_v201_list = param_all_norm_v201.tolist()
tmp_list = copy.deepcopy(param_all_norm_v201_list)

for idx in range(len(param_all_norm_v201_list)):
    tmp_list[idx].append(train_list[idx])

train_list_sorted = natsorted(tmp_list, key=lambda x: x[102])
sorted_param_all_norm_v201_list = [val[0:102] for val in train_list_sorted]
sorted_train_list = [val[102:103] for val in train_list_sorted]

sorted_param_all_norm_v201_list_np = np.asarray(sorted_param_all_norm_v201_list)
with open('3dmm_data/new_param_all_norm_v201.pkl', 'wb') as p:
    pickle.dump(sorted_param_all_norm_v201_list_np, p)

sorted_train_list = list(itertools.chain.from_iterable(sorted_train_list))
str_ = '\n'.join(sorted_train_list)
with open('3dmm_data/new_train_aug_120x120.list.train', 'wt') as f:
    f.write(str_)
