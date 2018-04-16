#! python3

# import plaidml.keras
# plaidml.keras.install_backend()

import numpy as np
from pathlib import Path

import keras
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
# from keras.layers import Input
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
# from keras.applications.vgg19 import VGG19
# from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input


# 学習データのパス
LABEL_FILE = 'label.txt'
DATASET_FILE = 'dataset.npz'
MODEL_NAME = 'model.json'
WEIGHT_DIR = 'weight'
WEIGHT_NAME = 'weight.hdf5' #'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
BATCH_SIZE = 128
EPOCH = 20


# ラベルのロード
label = []
with open(LABEL_FILE, 'r') as f:
    label = f.read().split('\n')

# データセットのロード
print('dataset load {}'.format(DATASET_FILE))
dataset = np.load(DATASET_FILE)
X_dataset = dataset['features']
y_dataset = dataset['labels']

# モデルのロード
model = None
if Path(MODEL_NAME).exists():
    # 前回のモデルがあればロードする
    print('model load {}'.format(MODEL_NAME))
    with open(MODEL_NAME, 'r') as f:
        model = model_from_json(f.read())

else:
    # 学習済みモデルのロード
    print('model not found.')
    # pretrained_model = InceptionV3(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
    # pretrained_model = VGG19(weights='imagenet')
    pretrained_model = ResNet50(weights='imagenet')

    # 中間層を出力するモデル
    # intermediate_layer_model = Model(inputs=pretrained_model.input, outputs=pretrained_model.layers[311].output)
    # intermediate_layer_model = Model(inputs=pretrained_model.input, outputs=pretrained_model.layers[21].output)
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
    open(MODEL_NAME, 'w').write(model_json)

# 一旦全レイヤーをフリーズ
for layer in model.layers:
    layer.trainable = False

# 最終段のDenseだけ再学習する
# model.layers[312].trainable = True
# model.layers[313].trainable = True
# model.layers[22].trainable = True
# model.layers[23].trainable = True
model.layers[176].trainable = True
model.layers[177].trainable = True

# 計算グラフのコンパイル
print('compile model.')
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# データセットを学習用とテスト用に分割
X_dataset = preprocess_input(X_dataset)
X_train, X_test, y_train, y_test = train_test_split(
    X_dataset, y_dataset, test_size=0.02, random_state=42)

# 学習済みの重みの取得
weights = sorted(Path(WEIGHT_DIR).glob('*'))
if len(weights) > 0:
    print('weight load {}'.format(weights[-1]))
    model.load_weights(weights[-1])

# 学習の実行
model_path = str(Path(WEIGHT_DIR).joinpath(WEIGHT_NAME))
checkpointer = keras.callbacks.ModelCheckpoint(model_path)
model.fit(X_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE,
          validation_data=(X_test, y_test),
          callbacks=[checkpointer])

# 評価
loss, acc = model.evaluate(X_test, y_test)
print('Loss {}, Accuracy {}'.format(loss, acc))


