#! python3

from progressbar import ProgressBar
from pathlib import Path
import numpy as np

import keras
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img

# 学習データの設定
DATA_DIR = 'dataset'
LABEL_FILE = 'label.txt'
SAVE_NAME = 'dataset.npz'
IMG_SIZE = 224

# ラベルの読み込み
LABEL_DATA = []
with open(LABEL_FILE, 'r') as f:
    LABEL_DATA = f.read().split('\n')

# 画像とラベルデータ
X = []
Y = []

# 対象画像の読み込みとラベリング
for index, label in enumerate(LABEL_DATA):
    path = Path(DATA_DIR).joinpath(label)
    pics = list_pictures(path)
    prog = ProgressBar(0, len(pics))
    print('[{}/{}] load {} {} pictures.'.format(index+1, len(LABEL_DATA), path, len(pics)))
    for i, picture in enumerate(pics):
        img = img_to_array(load_img(picture, target_size=(IMG_SIZE, IMG_SIZE)))
        X.append(img)
        Y.append(index)
        prog.update(i+1)

# arrayに変換
X = np.asarray(X)
Y = np.asarray(Y)

# 画素値を0から1の範囲に変換
X = keras.backend.cast_to_floatx(X)
X = X / 255.0

# クラスの形式を変換
Y = np_utils.to_categorical(Y, len(LABEL_DATA))

# データセットを保存
np.savez_compressed(SAVE_NAME, features=X, labels=Y)

