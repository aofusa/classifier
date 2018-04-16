#! python3

import sys

# 入力値チェック
if len(sys.argv) < 2:
    print('Usage: python3 {} filepath'.format(sys.argv[0]))
    quit()


import numpy as np
from pathlib import Path

from keras.preprocessing import image
from keras.models import model_from_json
# from keras.applications.inception_v3 import preprocess_input
# from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import preprocess_input


# データパス
LABEL_FILE = 'label.txt'
MODEL_NAME = 'model.json'
WEIGHT_DIR = 'weight'


# ラベルのロード
label = open(LABEL_FILE, 'r').read().split('\n')

# モデルのロード
model = model_from_json(open(MODEL_NAME, 'r').read())
model.load_weights(sorted(Path(WEIGHT_DIR).glob('*'))[-1])

# 画像から入力データを作成
for img_path in sys.argv[1:]:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 推論実行
    preds = model.predict(x)
    pred = dict(zip(label, preds[0]))
    ranking = sorted(pred.items(), key=lambda x: -x[1])

    # 推論結果を出力
    print('\nFile: {}'.format(img_path))
    print('Predicted: {}'.format(label[np.argmax(preds)]))
    for i, v in enumerate(ranking):
        print(i, v)
        if i > 5:
            break

