#! python3

import numpy as np
from pathlib import Path
import math
import argparse

# 引数のパーシング
parser = argparse.ArgumentParser(description='train network')
parser.add_argument('-d', '--dataset', help='path to dataset (default ./dataset.npz)')
parser.add_argument('-l', '--label', help='label text file (default ./label.txt)')
parser.add_argument('-m', '--model', help='save model json name (default ./model.json)')
parser.add_argument('-w', '--weight', help='save weight hdf5 name (default ./weight.hdf5)')
parser.add_argument('-o', '--output', help='save weight output directory (default ./weight)')
parser.add_argument('-b', '--batch', help='batch size (default 128)', type=int)
parser.add_argument('-e', '--epoch', help='num epochs (default 20)', type=int)
args = parser.parse_args()

# 学習データのパス
DATASET_FILE = 'dataset.npz'
LABEL_FILE = 'label.txt'
MODEL_NAME = 'model.json'
WEIGHT_NAME = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
WEIGHT_DIR = 'weight'
BATCH_SIZE = 128
EPOCH = 20

if args.dataset:
    DATASET_FILE = args.dataset
if args.label:
    LABEL_FILE = args.label
if args.model:
    MODEL_NAME = args.model
if args.weight:
    WEIGHT_NAME = args.weight
if args.output:
    WEIGHT_DIR = args.output
if args.batch:
    BATCH_SIZE = args.batch
if args.epoch:
    EPOCH = args.epoch

import keras
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import model_from_json
# from keras.layers import Input
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
# from keras.applications.vgg19 import VGG19
# from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

# ラベルのロード
label = []
with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    label = f.read().split('\n')

# データセットのロード
print('dataset load {}'.format(DATASET_FILE))
dataset = np.load(DATASET_FILE)
y_dataset = np_utils.to_categorical(dataset['labels'], len(label))
X_dataset = keras.backend.cast_to_floatx(dataset['features']) / 255.0
del dataset

# モデルのロード
model = None
if Path(MODEL_NAME).exists():
    # 前回のモデルがあればロードする
    print('model load {}'.format(MODEL_NAME))
    with open(MODEL_NAME, 'r', encoding='utf-8') as f:
        model = model_from_json(f.read())

else:
    # 学習済みモデルのロード
    print('model not found.')
    # pretrained_model = InceptionV3(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
    # pretrained_model = VGG19(weights='imagenet')
    pretrained_model = ResNet50(weights='imagenet')

    # 中間層を出力するモデル
    # intermediate_layer_model = Model(inputs=pretrained_model.input, outputs=pretrained_model.layers[311].output)
    # intermediate_layer_model = Model(inputs=pretrained_model.input, outputs=pretrained_model.layers[24].output)
    intermediate_layer_model = Model(inputs=pretrained_model.input, outputs=pretrained_model.layers[175].output)

    # Denseレイヤーを接続
    x = intermediate_layer_model.output
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(label), activation='softmax')(x)

    # 転移学習モデル
    model = Model(inputs=intermediate_layer_model.input, outputs=predictions)

    # モデルの書き出し
    print('save model {}'.format(MODEL_NAME))
    model_json = model.to_json()
    open(MODEL_NAME, 'w', encoding='utf-8').write(model_json)

# 一旦全レイヤーをフリーズ
for layer in model.layers:
    layer.trainable = False

# 最終段のDenseだけ再学習する
# model.layers[312].trainable = True
# model.layers[313].trainable = True
# model.layers[25].trainable = True
# model.layers[26].trainable = True
model.layers[176].trainable = True
model.layers[177].trainable = True

# 計算グラフのコンパイル
print('compile model.')
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# データセットを学習用とテスト用に分割
result = []
position = 0
for index, value in enumerate(y_dataset):
    if not (value == y_dataset[position]).all():
        result.append((position, index))
        position = index
    if index == len(y_dataset)-1:
        result.append((position, index+1))

train_length = 0
test_length = 0
for value in result:
    length = value[1] - value[0]
    spot = math.ceil(length * (1.0 - 0.1))
    train_length += spot
    test_length += (length-spot)
    y_dataset[train_length:len(y_dataset)] = np.roll(y_dataset[train_length:len(y_dataset)], -(length-spot), axis=0)
    X_dataset[train_length:len(X_dataset)] = np.roll(X_dataset[train_length:len(X_dataset)], -(length-spot), axis=0)

X_train, X_test = X_dataset[0:train_length], X_dataset[train_length:len(X_dataset)]
y_train, y_test = y_dataset[0:train_length], y_dataset[train_length:len(y_dataset)]
del X_dataset
del y_dataset

# 学習済みの重みの取得
weights = sorted(Path(WEIGHT_DIR).glob('*'))
if Path(WEIGHT_DIR, '.gitignore') in weights:
    weights.remove(Path(WEIGHT_DIR, '.gitignore'))
if len(weights) > 0:
    w = None
    if Path(WEIGHT_DIR, WEIGHT_NAME).exists():
        w = Path(WEIGHT_DIR).joinpath(WEIGHT_NAME)
    else:
        w = weights[-1]
    print('weight load {}'.format(w))
    model.load_weights(w)

# 学習の実行
model_path = str(Path(WEIGHT_DIR).joinpath(WEIGHT_NAME))
checkpointer = keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True)
model.fit(X_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE,
          validation_data=(X_test, y_test),
          callbacks=[checkpointer])

# 評価
loss, acc = model.evaluate(X_test, y_test)
print('Loss {}, Accuracy {}'.format(loss, acc))


