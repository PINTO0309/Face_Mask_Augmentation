import numpy as np
import pickle

with open('3dmm_data/new_param_all_norm_v201.pkl', 'rb') as p:
    param_all_norm_v201 = pickle.load(p)
print(f'param_all_norm_v201: {len(param_all_norm_v201)}')

new_param_all_norm_v201 = []
for val in param_all_norm_v201:
    new_param_all_norm_v201.append(val)
    new_param_all_norm_v201.append(val)

new_np_param_all_norm_v201 = np.asarray(new_param_all_norm_v201)
print(new_np_param_all_norm_v201.shape)

with open('3dmm_data/new_new_param_all_norm_v201.pkl', 'wb') as p:
    pickle.dump(new_np_param_all_norm_v201, p)

# new_np_param_all_norm_v201: 1272504 -> 636252 x2
print(f'new_np_param_all_norm_v201: {len(new_np_param_all_norm_v201)}')
