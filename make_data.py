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

import keras
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img

# ラベルの読み込み
LABEL_DATA = []
with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    LABEL_DATA = f.read().split('\n')

# 画像とラベルデータ
X = np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype='float32')
Y = np.empty(0, dtype=int)

# 対象画像の読み込みとラベリング
for index, label in enumerate(LABEL_DATA):
    path = Path(DATA_DIR).joinpath(label)
    pics = list_pictures(path)
    prog = ProgressBar(0, len(pics))
    Y = np.append(Y, np.full(len(pics), index, dtype=int))
    print('[{}/{}] load {} {} pictures.'.format(index+1, len(LABEL_DATA), path, len(pics)))
    for i, picture in enumerate(pics):
        img = img_to_array(load_img(picture, target_size=(IMG_SIZE, IMG_SIZE)))
        img = keras.backend.cast_to_floatx(img) / 255.0  # 画素値を0から1の範囲に変換
        img = np.expand_dims(img, axis=0)
        X = np.append(X, img, axis=0)
        prog.update(i+1)

# クラスの形式を変換
Y = np_utils.to_categorical(Y, len(LABEL_DATA))

# データセットを保存
np.savez_compressed(SAVE_NAME, features=X, labels=Y)

