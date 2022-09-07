from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from model import DropcastDataset
from model import FCN
from model import train_model
from model import get_predictions
from model import plot_predictions


def main():
    # Set seeds for reproducibility
    np.random.seed(2408)
    torch.manual_seed(2408)

    # Set hyperparameters
    device = 'cuda:0'
    date = '20220829'
    learningRate = 5e-4
    momentum = 0.9
    decay = 1e-3
    batchSize = 32
    nEpochs = 20
    criterion = [nn.CrossEntropyLoss(), nn.MSELoss()]

    # Load the dataset
    print('Loading data...')

    # Get sample usage from file
    allSamples = pd.read_excel('MeasuredSamples.xlsx', sheet_name='List')

    trainSamples = allSamples.loc[allSamples['Usage'] == 'training',
                                  'Sample'].values
    validSamples = allSamples.loc[allSamples['Usage'] == 'validation',
                                  'Sample'].values
    print(f'Using {len(trainSamples)} training samples.')
    print(f'Using {len(validSamples)} validation samples.')

    # Initialize datasets
    trainSet = DropcastDataset(
        Path('data/'),
        trainSamples,
        maxTime=60 * 120)
    validSet = DropcastDataset(
        Path('data/'),
        validSamples,
        maxTime=trainSet.maxTime,
        catTemps=trainSet.categories)

    trainSetSize = len(trainSet)
    validSetSize = len(validSet)
    print(f'Training set has {trainSetSize} images.')
    print(f'Validation set has {validSetSize} images.')

    # Create performance optimized DataLoaders
    trainingLoader = torch.utils.data.DataLoader(
        trainSet, batch_size=batchSize, shuffle=True,
        num_workers=6, pin_memory=True)
    validationLoader = torch.utils.data.DataLoader(
        validSet, batch_size=64,
        num_workers=6, pin_memory=True)

    # Define the model and loss
    model = FCN(maxTime=trainSet.maxTime, categories=trainSet.categories)
    if 'cuda' in device:
        model = model.to(device)

    # Create and initialize TensorBoard writer
    name = f'{date}_LR{learningRate:.1e}_momentum{momentum:.1e}_'\
        f'decay{decay:.1e}_batch{batchSize}_{len(trainSamples)}samples'
    writer = SummaryWriter(f'logs/{name}')
    print(f'This is {name} running on {device}')

    # Grab a single mini-batch of inputs
    dataiter = iter(trainingLoader)
    _, inputs, _ = dataiter.next()
    inputs = inputs.to(device)

    # Trace the sample input through your model, and render it as a graph
    writer.add_graph(model, inputs)
    writer.flush()

    # Train the model
    print('Training started')
    train_model(trainingLoader, validationLoader, model, criterion[0],
                criterion[1], writer, lr=learningRate, momentum=momentum,
                lamb=decay, nesterov=True, nEpochs=nEpochs, device=device)
    print('Training finished')
    writer.flush()

    # Save model
    torch.save(model, f'{name}.pt')

    # Plot final predictions for validation
    predictions = get_predictions(validSet, model, device)
    plot_predictions(predictions, name)
    predictions = get_predictions(trainSet, model, device)
    plot_predictions(predictions, name + '_training')


if __name__ == '__main__':
    main()
