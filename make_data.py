#! python3

from progressbar import ProgressBar
from pathlib import Path
import argparse
import numpy as np

# 引数のパーシング
parser = argparse.ArgumentParser(description='make dataset')
parser.add_argument('-d', '--dir', help='path to dataset directory (default ./dataset)')
parser.add_argument('-l', '--label', help='label text file (default ./label.txt)')
parser.add_argument('-o', '--output', help='save numpy file (.npz) name (default ./dataset.npz)')
parser.add_argument('-s', '--size', help='resizze image (default 224)', type=int)
args = parser.parse_args()

# 学習データの設定
DATA_DIR = './dataset'
LABEL_FILE = './label.txt'
SAVE_NAME = './dataset.npz'
IMG_SIZE = 224

if args.dir:
    DATA_DIR = args.dir
if args.label:
    LABEL_FILE = args.label
if args.output:
    SAVE_NAME = args.output
if args.size:
    IMG_SIZE = args.size

from keras.preprocessing.image import img_to_array, list_pictures, load_img

# ラベルの読み込み
LABEL_DATA = []
with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    LABEL_DATA = f.read().split('\n')

# 全画像枚数
NUM_PICS = 0
print('count pictures.')
prog = ProgressBar(0, len(LABEL_DATA))
for index, label in enumerate(LABEL_DATA):
    path = Path(DATA_DIR).joinpath(label)
    pics = list_pictures(path)
    NUM_PICS += len(pics)
    prog.update(index+1)
prog.finish()
print('{} pictures.'.format(NUM_PICS))

# 画像とラベルデータ
print('reserve memory {} byte.'.format(NUM_PICS*IMG_SIZE*IMG_SIZE*3*1))
Y = np.empty(NUM_PICS, dtype=int)
X = np.empty((NUM_PICS, IMG_SIZE, IMG_SIZE, 3), dtype='int8')

# 対象画像の読み込みとラベリング
IMG = np.empty((1, IMG_SIZE, IMG_SIZE, 3), dtype='int8')
POSITION = 0
ERROR_COUNT = 0
for index, label in enumerate(LABEL_DATA):
    path = Path(DATA_DIR).joinpath(label)
    pics = list_pictures(path)
    prog = ProgressBar(0, len(pics))
    print('[{}/{}] load {} {} pictures.'.format(index+1, len(LABEL_DATA), path, len(pics)))
    count = 0
    for picture in pics:
        try:
            IMG[0] = img_to_array(load_img(picture, target_size=(IMG_SIZE, IMG_SIZE)))
            X[POSITION+count] = IMG
            prog.update(count+1)
            count += 1
        except Exception as identifier:
            print(picture, identifier)
            pics.remove(picture)
            ERROR_COUNT += 1
    Y[POSITION:POSITION+len(pics)] = np.full(len(pics), index, dtype=int)
    POSITION += len(pics)
    prog.finish()

# データセットを保存
if ERROR_COUNT > 0:
    np.savez_compressed(SAVE_NAME, features=X[:-ERROR_COUNT], labels=Y[:-ERROR_COUNT])
else:
    np.savez_compressed(SAVE_NAME, features=X, labels=Y)

