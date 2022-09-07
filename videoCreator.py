import cv2
import os
from datetime import datetime

for temperature in os.listdir('data/'):
    if len(temperature) > 3:
        continue
    print('Temperature:', temperature)
    for sample in os.listdir('data/' + temperature):
        print('Sample', sample)
        image_folder = f'data/{temperature}/{sample}'
        video_name = f'videos/{temperature}/{sample}.avi'

        if os.path.isfile(video_name):
            continue

        images = [img for img in os.listdir(image_folder) if img.endswith('.webp')]
        images = sorted(images)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 30, (width, height))

        for i, image in enumerate(images):
            _, date, time = image.rstrip('.webp').split('_')
            datetime = datetime.strptime(
                ' '.join([date, time]), '%Y%m%d %H%M%S')
            if i == 0:
                start = datetime
            elapsed = datetime - start
            img = cv2.imread(os.path.join(image_folder, image))
            img = cv2.putText(img,
                              f'{temperature} degC, {elapsed}',
                              (5, height - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (255, 255, 255),
                              2)
            video.write(img)
        print(f'Measurement time: {elapsed}.')
        cv2.destroyAllWindows()
        video.release()
    print('- - - - - - - - - - - -')
print('All samples finished')
