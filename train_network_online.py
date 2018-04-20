#! python3

import argparse

# 引数のパーシング
parser = argparse.ArgumentParser(description='train network')
parser.add_argument('-d', '--dataset', help='path to dataset (default ./dataset)')
parser.add_argument('-l', '--label', help='label text file (default ./label.txt)')
parser.add_argument('-m', '--model', help='save model json name (default ./model.json)')
parser.add_argument('-w', '--weight', help='save weight hdf5 name (default weight.hdf5)')
parser.add_argument('-o', '--output', help='save weight output directory (default ./weight)')
parser.add_argument('-b', '--batch', help='batch size (default 128)', type=int)
parser.add_argument('-e', '--epoch', help='num epochs (default 20)', type=int)
parser.add_argument('-s', '--sample', help='num samples (default 6600)', type=int)
args = parser.parse_args()


# 学習データの設定
DATA_DIR = './dataset'
LABEL_FILE = './label.txt'
MODEL_NAME = './model.json'
WEIGHT_NAME = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
WEIGHT_DIR = './weight'
IMG_SIZE = 224

# 学習パラメータ
BATCH_SIZE = 128
EPOCH = 20
SAMPLING = 6600

if args.dataset:
    DATA_DIR = args.dataset
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
if args.sample:
    SAMPLING = args.sample


from progressbar import ProgressBar
from pathlib import Path
import math
import numpy as np
import keras
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, list_pictures, load_img
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

# ラベルの読み込み
LABEL_DATA = []
with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    LABEL_DATA = f.read().split('\n')

# 全画像枚数
NUM_PICS = 0
LIST_PICS = []
print('count pictures.')
prog = ProgressBar(0, len(LABEL_DATA))
for index, label in enumerate(LABEL_DATA):
    path = Path(DATA_DIR).joinpath(label)
    pics = list_pictures(path)
    for p in pics:
        LIST_PICS.append((p, index))
    NUM_PICS += len(pics)
    prog.update(index+1)
prog.finish()
LIST_PICS = np.asarray(LIST_PICS)
print('{} pictures.'.format(NUM_PICS))

# 画像とラベルデータ
print('reserve memory {} byte.'.format(SAMPLING*IMG_SIZE*IMG_SIZE*3*4))
Y = np.empty(SAMPLING, dtype=int)
X = np.empty((SAMPLING, IMG_SIZE, IMG_SIZE, 3), dtype=keras.backend.floatx())
IMG = np.empty((1, IMG_SIZE, IMG_SIZE, 3), dtype=keras.backend.floatx())

# モデルのロード
model = None
if Path(MODEL_NAME).exists():
    # 学習済みモデルのロード
    print('model load {}'.format(MODEL_NAME))
    with open(MODEL_NAME, 'r', encoding='utf-8') as f:
        model = model_from_json(f.read())
else:
    # モデルの作成
    print('model not found.')
    pretrained_model = ResNet50(weights='imagenet')
    intermediate_layer_model = Model(inputs=pretrained_model.input, outputs=pretrained_model.layers[175].output)
    x = intermediate_layer_model.output
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(label), activation='softmax')(x)
    model = Model(inputs=intermediate_layer_model.input, outputs=predictions)
    print('save model {}'.format(MODEL_NAME))
    model_json = model.to_json()
    open(MODEL_NAME, 'w', encoding='utf-8').write(model_json)

# 一旦全レイヤーをフリーズ
for layer in model.layers:
    layer.trainable = False

# 最終段のDenseだけ再学習する
model.layers[176].trainable = True
model.layers[177].trainable = True

# 計算グラフのコンパイル
print('compile model.')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 学習済みの重みの取得
weights = sorted(Path(WEIGHT_DIR).glob('*'))
if len(weights) > 0:
    print('weight load {}'.format(weights[-1]))
    model.load_weights(weights[-1])

# オンライン学習
for epoch in range(0, EPOCH):
    # サンプリングする画像リストのシャッフル
    np.random.shuffle(LIST_PICS)

    for sample in range(0, math.ceil(NUM_PICS/SAMPLING)):
        # 対象画像の読み込みとラベリング
        ERROR_COUNT = 0
        END = SAMPLING*(sample+1)
        if END > len(LIST_PICS):
            END = len(LIST_PICS)
        prog = ProgressBar(0, END)
        print('[{}/{}] load {} pictures.'.format(sample+1, math.ceil(NUM_PICS/SAMPLING), END))
        count = 0
        for index, picture in enumerate(LIST_PICS[SAMPLING*sample:END]):
            try:
                IMG[0] = img_to_array(load_img(picture[0], target_size=(IMG_SIZE, IMG_SIZE)))
                IMG /= 255.0
                X[count] = IMG
                Y[count] = picture[1]
                prog.update(count+1)
                count += 1
            except Exception as identifier:
                print(picture, identifier)
                pics.remove(picture)
                ERROR_COUNT += 1
        prog.finish()

        # データセットを学習用とテスト用に分割
        X_dataset = preprocess_input(X[:END-ERROR_COUNT])
        y_dataset = np_utils.to_categorical(Y[:END-ERROR_COUNT], len(LABEL_DATA))
        np.random.seed(42 + epoch)
        np.random.shuffle(X_dataset)
        np.random.seed(42 + epoch)
        np.random.shuffle(y_dataset)
        spot = math.ceil(X.shape[0] * 0.1)
        X_train, X_test, y_train, y_test = X_dataset[spot:], X_dataset[:spot], Y[spot:], Y[:spot]

        # 学習の実行
        model_path = str(Path(WEIGHT_DIR).joinpath(WEIGHT_NAME))
        checkpointer = keras.callbacks.ModelCheckpoint(model_path)
        model.fit(X_train, y_train, epochs=epoch+1, batch_size=BATCH_SIZE,
                initial_epoch=epoch, validation_data=(X_test, y_test),
                callbacks=[checkpointer])

        # 評価
        loss, acc = model.evaluate(X_test, y_test)
        print('Loss {}, Accuracy {}'.format(loss, acc))


