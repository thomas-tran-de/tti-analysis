from pathlib import Path
import torch
import torch.utils.data
import pandas as pd

from model import DropcastDataset
from model import FCN
from model import get_predictions
from model import plot_predictions

# Load model
name = 'FinalModel'
model = torch.load('ModelWeights.pt')

if torch.cuda.is_available():
    model = model.to(device)
model.eval()

# Load dataset
allSamples = pd.read_excel('MeasuredSamples.xlsx', sheet_name='List')
trainSamples = allSamples.loc[allSamples['Usage'] == 'training',
                              'Sample'].values
validSamples = allSamples.loc[allSamples['Usage'] == 'validation',
                              'Sample'].values

print(f'Training set has {trainSamples.size} images.')
print(f'Validation set has {validSamples.size} images.')

trainSet = DropcastDataset(
    Path('data/'),
    trainSamples,
    maxTime=model.maxTime,
    catTemps=model.categories)
dataSet = DropcastDataset(
    Path('data/'),
    validSamples,
    maxTime=model.maxTime,
    catTemps=model.categories)

print('Creating predictions...')
predictions = get_predictions(dataSet, model, device)
predictions.to_csv(f'{name}_predictions.csv', index=False)
# predictions = pd.read_csv(f'{name}_predictions.csv')
plot_predictions(predictions, name)
predictions = get_predictions(trainSet, model, device)
predictions.to_csv(f'{name}_predictions_training.csv', index=False)
plot_predictions(predictions, name + '_training')
