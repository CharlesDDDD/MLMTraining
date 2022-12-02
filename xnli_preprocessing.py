import glob

import numpy as np
import pandas as pd

listOffiles = glob.glob('data/*.tsv')
text = np.array([])
print(listOffiles)
for file in listOffiles:
    data = pd.read_csv(file, sep='\t')
    col1 = data.iloc[:, 0].values
    col2 = data.iloc[:, 1].values
    data_pro = np.concatenate([col1, col2])
    text = np.concatenate([text, data_pro])

print(text.shape)
print(text)
np.save('data/dev-all.npy', text)
