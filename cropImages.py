from pathlib import Path
from PIL import Image as PImage
from datetime import datetime
import pandas as pd


class Image():
    def __init__(self, pathToFile, x, y, width, height):
        """
        """
        if isinstance(pathToFile, str):
            pathToFile = Path(pathToFile)
        _, date, time = pathToFile.stem.split('_')
        self.datetime = datetime.strptime(
            ' '.join([date, time]), '%Y%m%d %H%M%S')

        img = PImage.open(pathToFile)
        img = img.crop((x, y, width + x, height + y))
        self._img = img

    def show(self):
        return self._img.show()

    def save(self, fileName):
        return self._img.save(fileName)


if __name__ == '__main__':
    allNames = []
    allTemps = []

    for folder in Path(f'G:/TMM_ordered/').iterdir():
        if folder.is_file():
            continue
        temperature = folder.name
        for subfolder in folder.iterdir():
            sample = subfolder.name
            if 'Bad' in sample:
                continue

            x = 1925
            y = 830
            height = 420
            width = 1060

            saveFolder = Path('data') / temperature / sample
            try:
                saveFolder.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue

            allTemps.append(temperature)
            allNames.append(sample)
            print(f'Processing {sample}')

            fileNames = []
            timestamps = []
            potentialFiles = list(subfolder.iterdir())
            for i, file in enumerate(potentialFiles, 1):
                if file.suffix != '.webp':
                    continue
                img = Image(file, x, y, width, height)
                img.save(saveFolder / (file.stem + '.webp'))
                fileNames.append(str(saveFolder / (file.stem + '.webp')))
                timestamps.append(img.datetime)
                if i % 300 == 0:
                    print(f'Processed {i} of {len(potentialFiles)} files')
            print(f'Processed all files for {sample}')
            startTime = min(timestamps)
            times = [(t - startTime).seconds for t in timestamps]
            labelData = pd.DataFrame(
                {'Sample': len(fileNames) * [sample],
                 'File': fileNames,
                 'Temperature / °C': len(fileNames) * [temperature],
                 'Time / s': times})

            try:
                labelFile = pd.read_csv(folder.parent / 'Labels.csv')
                labelFile = pd.concat([labelData, labelFile])
            except FileNotFoundError:
                labelFile = labelData
            labelFile.to_csv(folder.parent / 'Labels.csv', index=False)
            print('Labels appended')

    df = pd.DataFrame({'Samples': allNames, 'Temperatures': allTemps})
    df.to_csv('ProcessedSamples.csv')
    print('Finished all samples')
