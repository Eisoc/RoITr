import pickle



with open('configs/tdmatch/train_info.pkl', 'rb') as f:
    my_object = pickle.load(f)

with open('configs/tdmatch/val_info.pkl', 'rb') as f:
    my_object2 = pickle.load(f)

with open('configs/tdmatch/3DLoMatch.pkl', 'rb') as f:
    my_object3 = pickle.load(f)

print("finished")