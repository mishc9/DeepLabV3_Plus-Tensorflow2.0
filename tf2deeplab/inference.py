from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tf2deeplab.deeplab import DeepLabV3Plus
from tqdm import tqdm
import os
from glob import glob
import pickle
from moviepy.editor import ImageSequenceClip

from tf2deeplab.utils import pipeline

h, w = 800, 1600
with open('../notebooks/cityscapes_dict.pkl', 'rb') as f:
    id_to_color = pickle.load(f)['color_map']

model = DeepLabV3Plus(h, w, 34)
model.load_weights('top_weights.h5')

image_dir = '/home/mia/backup/research/autonomous_driving/cityscapes/dataset/val_images'
image_list = os.listdir(image_dir)
image_list.sort()
print(f'{len(image_list)} frames found')


test = load_img(f'{image_dir}/{image_list[1]}')
test = img_to_array(test)
pipeline(test, video=False)

for image_dir in ['stuttgart_00', 'stuttgart_01', 'stuttgart_02']:
    os.mkdir(f'outputs/{image_dir}')
    image_list = os.listdir(image_dir)
    image_list.sort()
    print(f'{len(image_list)} frames found')
    for i in tqdm(range(len(image_list))):
        try:
            test = load_img(f'{image_dir}/{image_list[i]}')
            test = img_to_array(test)
            segmap = pipeline(test, video=False,
                              fname=f'{image_list[i]}', folder=image_dir)
            if segmap == False:
                break
        except Exception as e:
            print(str(e))
    clip = ImageSequenceClip(
        sorted(glob(f'outputs/{image_dir}/*')), fps=18, load_images=True)
    clip.write_videofile(f'{image_dir}.mp4')
