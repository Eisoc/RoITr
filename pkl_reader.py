import pickle



with open('configs/tdmatch/train_info.pkl', 'rb') as f:
    train_info = pickle.load(f)

with open('configs/tdmatch/val_info.pkl', 'rb') as f:
    val_info = pickle.load(f)

with open('configs/tdmatch/3DLoMatch.pkl', 'rb') as f:
    pkl_3DLoMatch = pickle.load(f)
with open('configs/tdmatch/9675.pkl', 'rb') as f:
    pkl_9675 = pickle.load(f)

print("finished")