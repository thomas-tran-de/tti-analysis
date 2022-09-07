import pandas as pd
from pathlib import Path

inputs = pd.read_csv('data/Inputs.csv', converters={'File': Path},
                     index_col='File')

means = inputs.mean()
std = inputs.std()
stats = pd.concat([means, std], axis=1)
stats.to_csv('data/Stats.csv', header=['Mean', 'Std'])
