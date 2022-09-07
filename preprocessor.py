import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path


def get_means(image):
    cutSize = 75
    cutPositions = [
        (27, 16), (128, 16), (227, 15), (334, 16), (439, 18),
        (544, 22), (646, 20), (753, 23), (854, 20), (959, 22),
        (23, 116), (125, 115), (229, 114), (333, 117), (439, 113),
        (540, 119), (647, 120), (753, 121), (855, 121), (961, 121),
        (22, 209), (124, 212), (227, 214), (334, 214), (441, 217),
        (544, 217), (652, 219), (756, 220), (861, 223), (967, 224),
        (20, 314), (125, 317), (227, 314), (334, 317), (441, 316),
        (546, 321), (652, 322), (759, 322), (863, 322), (968, 323)]
    means = []
    for x, y in cutPositions:
        crop = image.crop([x, y, x + cutSize, y + cutSize])
        arr = np.array(crop)
        arr = arr[:, :, 0].flatten()
        arr.sort()
        size = arr.size
        perc = size // 100
        means.append(arr[-5 * perc:].mean())
        means.append(arr[-10 * perc:].mean())
        means.append(arr[-15 * perc:].mean())
        means.append(arr[-20 * perc:].mean())
        means.append(arr[-25 * perc:].mean())
    return means


if __name__ == '__main__':
    dataFolder = Path('data/')
    processed = pd.read_csv(dataFolder / 'Inputs.csv',
                            converters={'File': Path}, index_col='File')
    labels = pd.read_csv(dataFolder / 'Labels.csv',
                         converters={'File': Path}, index_col='File')
    oldSamples = processed.join(labels, how='inner')['Sample'].unique()
    allFiles = []
    allValues = []
    for temperature in dataFolder.iterdir():
        if temperature.is_file():
            continue
        print('Temperature:', temperature)
        for sample in temperature.iterdir():
            if sample.name in oldSamples:
                continue
            print('Sample:', sample)
            for file in sample.iterdir():
                img = Image.open(file)
                means = get_means(img)
                allValues.append(means)
                allFiles.append(file)
        print('- - - - - - - - - -')

    names = ['top 5', 'top 10', 'top 15', 'top 20', 'top 25']
    columns = [f'Spot {i}, {n}' for i in range(1, 41) for n in names]
    df = pd.DataFrame(allValues, index=allFiles, columns=columns)
    df.index.name = 'File'
    df = pd.concat([processed, df])
    df.to_csv(dataFolder / 'Inputs.csv')
