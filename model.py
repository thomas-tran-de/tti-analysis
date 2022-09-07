import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path


class DropcastDataset(torch.utils.data.Dataset):
    def __init__(self, root, samples=None, maxTime=None, catTemps=None):
        """
        Create a DropcastDataset containing red value inputs and labels

        Parameters
        ----------
        root : pathlib.Path
            Directory containing the inputs, means for normalization,
            and labels
        samples : list of str, optional
            Use only the names of the given samples in the dataset.
            If not given, use all samples
        maxTime : float, optional
            Maximum time used for normalization. If not given, use all times
        catTemps : dict, optional
            The keys for temperature categories with the corresponding
            temperature. If not given, create a dictionary from the labels
        """
        self.root = root

        # Load input data
        norm = pd.read_csv(root / 'Stats.csv', index_col=0)
        inputs = pd.read_csv(root / 'Inputs.csv', index_col='File',
                             converters={'File': Path}, dtype='float32')
        for col in inputs.columns:
            inputs[col] = (inputs[col] - norm.loc[col, 'Mean']) / \
                norm.loc[col, 'Std']
        self.inputs = inputs

        # Load labels
        labels = pd.read_csv(root / 'Labels.csv', converters={'File': Path},
                             dtype={'Sample': str,
                                    'Temperature / °C': 'float32',
                                    'Time / s': 'float32'})

        # Use only given sample names
        if samples is not None:
            labels = labels[labels['Sample'].isin(samples)].copy()

        # Don't use the first 2 minutes due to temperature equilibration
        labels['Time / s'] -= 60 * 2

        # Normalize time to be between 0 and 1
        if maxTime:
            self.maxTime = maxTime
        else:
            self.maxTime = labels['Time / s'].max()
        labels['Time / s'] /= self.maxTime

        # Categorize temperature
        if catTemps is not None:
            categories = {v: k for k, v in catTemps.items()}
        else:
            categories = labels['Temperature / °C'].unique()
            categories.sort()
            categories = {k: v for v, k in enumerate(categories)}
        self.categories = {v: k for k, v in categories.items()}
        labels['Category'] = labels['Temperature / °C'].map(categories)

        self.labels = labels.loc[labels['Time / s'].between(0, 1)].reset_index()

    def __getitem__(self, idx):
        labels = self.labels.loc[idx]
        values = self.inputs.loc[labels['File']]
        labels = torch.tensor(labels[['Category', 'Time / s']])
        values = torch.tensor(values.values)
        return idx, values, labels

    def __len__(self):
        return len(self.labels.index)


class FCN(nn.Module):
    def __init__(self, maxTime, categories):
        super(FCN, self).__init__()

        self.maxTime = maxTime
        self.categories = categories

        inputSize = 200  # 40 spots with 5 red values each

        self.tempNet = nn.Sequential(
            nn.BatchNorm1d(inputSize),
            nn.Linear(inputSize, 8192),
            nn.LeakyReLU(),
            nn.Linear(8192, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(categories)),
            nn.Softmax(dim=1)
        )

        self.timeNet = nn.Sequential(
            nn.BatchNorm1d(inputSize + len(categories)),
            nn.Linear(inputSize + len(categories), 8192),
            nn.LeakyReLU(),
            nn.Linear(8192, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        temp = self.tempNet(x)
        x = torch.hstack([x, temp])
        time = self.timeNet(x)
        return temp, time


def train_model(trainData, validData, model, tempCrit, timeCrit, writer,
                lr=0.01, momentum=0.9, lamb=1e-3, nesterov=False, nEpochs=30,
                device='cpu'):
    """
    Train a model for N epochs given data and hyper-params with stochastic
    gradient descent

    Parameters
    ----------
    trainData : torch.utils.data.DataLoader
        DataLoader containing the training data with labels
    validData : torch.utils.data.DataLoader
        DataLoader containing the validation data with labels
    model : FCN
        Model to be trained
    tempCrit : torch.nn._Loss
        Loss function for the temperature output
    timeCrit : torch.nn._Loss
        Loss function for the time output
    writer : torch.utils.tensorboard.SummaryWriter
        Writer to log the training and validation loss
    lr : float, optional
        Learning rate
    momentum : float, optional
        Momentum factor
    lamb : float, optional
        Weight decay (L2 penalty)
    nesterov : bool, optional
        Enables Nesterov momentum
    nEpochs : int, optional
        Number of epochs to train
    device : str, optional
        Whether to use CPU ('cpu') or GPU ('cuda')
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=lamb, nesterov=nesterov)

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True

    batchSize = trainData.batch_size

    for epoch in range(nEpochs):
        print(f'Epoch {epoch + 1} of {nEpochs}')
        epochStartTime = time.time()
        runningLoss = 0.0

        for i, data in enumerate(trainData):
            # basic training loop
            _, inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            lblTemp = labels[:, 0].type(torch.LongTensor)
            lblTime = labels[:, 1].view(-1, 1)
            lblTemp = lblTemp.to(device, non_blocking=True)
            lblTime = lblTime.to(device, non_blocking=True)
            outTemp, outTime = model(inputs)
            lossTemp = tempCrit(outTemp, lblTemp)
            lossTime = timeCrit(outTime, lblTime)
            loss = lossTemp + lossTime

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            assert torch.isfinite(loss), f'Loss is {loss.item()}'

            runningLoss += loss.item()
            nRep = int(500 * (32 / batchSize))
            if i % nRep == (nRep - 1):    # Every nRep mini-batches...
                # ...check against the validation set
                print(f'Batch {i + 1} of {len(trainData)}')
                runningVLoss = 0.0
                runningTime = 0.0
                runningTemp = 0.0

                model.train(False)  # Turn off gradients for validation
                for j, vData in enumerate(validData):
                    _, vInputs, vLabels = vData
                    vInputs = vInputs.to(device, non_blocking=True)
                    vLblTemp = vLabels[:, 0].type(torch.LongTensor)
                    vLblTime = vLabels[:, 1].view(-1, 1)
                    vLblTemp = vLblTemp.to(device, non_blocking=True)
                    vLblTime = vLblTime.to(device, non_blocking=True)
                    vOutTemp, vOutTime = model(vInputs)
                    vLossTemp = tempCrit(vOutTemp, vLblTemp)
                    vLossTime = timeCrit(vOutTime, vLblTime)
                    vLoss = vLossTemp + vLossTime
                    runningTime += vLossTime.item()
                    runningTemp += vLossTemp.item()
                    runningVLoss += vLoss.item()
                model.train(True)  # Turn gradients back on for training

                avgLoss = runningLoss / (nRep - 1)
                avgVLoss = runningVLoss / len(validData)
                runningTime /= len(validData)
                runningTemp /= len(validData)

                # Log the running loss averaged per batch
                writer.add_scalars(
                    'Loss',
                    {'Training': avgLoss, 'Validation': avgVLoss,
                     'Time': runningTime, 'Temperature': runningTemp},
                    epoch * len(trainData) + i)

                runningLoss = 0.0

        print('Epoch {} took {:.0f} s.'.format(
            epoch + 1, time.time() - epochStartTime))
    model.train(False)


def get_predictions(dataSet, model, device):
    """ Create a DataFrame with predictions of the given model """
    dataLoader = torch.utils.data.DataLoader(
        dataSet, batch_size=64, shuffle=False)

    # Get correct labels and predictions
    allTemps = []
    allTimes = []
    allIdx = []
    for j, data in enumerate(dataLoader):
        if j % 50 == 0:
            print(f'Processing batch {j} of {len(dataLoader)}')
        idx, inputs, _ = data
        inputs = inputs.to(device, non_blocking=True)
        outTemp, outTime = model(inputs)

        predTemp = torch.argmax(outTemp, dim=1)
        predTemp = predTemp.detach().to('cpu').numpy()
        predTime = outTime.detach().to('cpu').numpy().flatten()

        allIdx += idx
        allTemps += list(predTemp)
        allTimes += list(predTime)

    df = dataSet.labels.loc[allIdx].copy()
    df['Predicted temperature / °C'] = [model.categories[c] for c in allTemps]
    df['Predicted time / s'] = allTimes
    df['Predicted time / s'] = df['Predicted time / s'] * model.maxTime + 120
    df['Time / s'] = df['Time / s'] * model.maxTime + 120

    return df


def plot_predictions(df, title):
    """ Plot labels vs predictions for a given DataFrame """
    maxTime = df['Time / s'].max() / 60  # Used for diagonal lines

    # Prepare 2 coordinate systems
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig, height_ratios=(2, 1),
                  wspace=0.08)
    axTime = fig.add_subplot(gs[0, 0])
    resTime = fig.add_subplot(gs[1, 0], sharex=axTime)

    axTime.plot([0, maxTime], [0, maxTime], c='k')
    resTime.axhline(0, c='k')

    df['Residuals / s'] = df['Predicted time / s'] - df['Time / s']

    # Plot absolute values and residuals
    sns.scatterplot(
        x=df['Time / s'] / 60,
        y=df['Predicted time / s'] / 60,
        hue=df['Predicted temperature / °C'], palette='plasma',
        ax=axTime, marker='.', ec='none', legend='full')
    sns.scatterplot(
        x=df['Time / s'] / 60,
        y=df['Residuals / s'] / 60,
        hue=df['Predicted temperature / °C'], palette='plasma',
        ax=resTime, marker='.', ec='none', legend=False)

    # Show mean, median, and standard deviation
    median = df['Residuals / s'].median() / 60
    mean = df['Residuals / s'].mean() / 60
    std = df['Residuals / s'].std() / 60
    lbl = f'Median: {median:.1f} min\nMean: {mean:.1f} min\nStd: {std:.1f} min'
    resTime.legend('', '', loc='upper left', bbox_to_anchor=(1, 1),
                   title=lbl)

    # Finalize figure
    fig.suptitle(title)
    sns.move_legend(axTime, 'center left', bbox_to_anchor=(1, 0.5))
    axTime.set_xlabel('')
    axTime.tick_params(labelbottom=False)
    axTime.set_ylabel('Predicted time / min')
    resTime.set_xlabel('Correct time / min')
    resTime.set_ylabel('Time residual / min')
    axTime.xaxis.set_major_locator(mticker.MultipleLocator(30))
    fig.savefig(f'{title}_time.png', dpi=400, bbox_inches='tight')

    # Prepare 2 coordinate systems
    fig = plt.figure(constrained_layout=True, figsize=[9, 4.5])
    axTemp, axTime = fig.subplots(1, 2)

    # Plot temperature predition as bar chart
    temps = range(100, 145, 5)
    bins = [t - 2 for t in temps] + [t + 2 for t in temps]
    bins.sort()
    axTemp = sns.histplot(
        data=df, x='Temperature / °C', hue='Predicted temperature / °C',
        stat='percent', palette='plasma', multiple='stack',
        bins=bins, ax=axTemp, legend='full')
    sns.move_legend(axTemp, 'center left', bbox_to_anchor=(1, 0.5))

    # Plot time predition as contour lines
    axTime = sns.kdeplot(
        x=df['Time / s'] / 60, y=df['Predicted time / s'] / 60,
        fill=True, ax=axTime, cbar=True, cbar_kws={'location': 'top'})
    axTime.plot([0, maxTime], [0, maxTime], c='k')
    axTime.set_aspect('equal')

    # Finalize figure
    fig.suptitle(title)
    fig.savefig(f'{title}.png', dpi=400, bbox_inches='tight')

    # Create figure with time preditctions as an array of plots
    def const_line(maxTime, **kwargs):
        x = (0, maxTime)
        plt.plot(x, x, **kwargs)

    fcGrid = sns.relplot(
        x=df['Time / s'] / 60,
        y=df['Predicted time / s'] / 60,
        hue=df['Predicted temperature / °C'], palette='plasma',
        col=df['Temperature / °C'], col_wrap=3,
        marker='.', ec='none', legend='full')
    (fcGrid.map(const_line, maxTime=maxTime, color='k', dashes=(2, 1), zorder=0)
     .set_titles('True temperature: {col_name} °C')
     .tight_layout(w_pad=0))
    plt.savefig(f'{title}_timeArray.png')

    plt.close('all')
