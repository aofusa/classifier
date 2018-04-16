#! python3

import numpy as np
from pathlib import Path
import argparse

# 引数のパーシング
parser = argparse.ArgumentParser(description='predict image')
parser.add_argument('filepath', help='image file path', nargs='+')
parser.add_argument('-l', '--label', help='label text file (default ./label.txt)')
parser.add_argument('-m', '--model', help='load model json (default ./model.json)')
parser.add_argument('-w', '--weight', help='load weight directory (default ./weight)')
args = parser.parse_args()

# データパス
LABEL_FILE = 'label.txt'
MODEL_NAME = 'model.json'
WEIGHT_DIR = 'weight'

if args.label:
    LABEL_FILE = args.label
if args.model:
    MODEL_NAME = args.model
if args.weight:
    WEIGHT_DIR = args.weight

from keras.preprocessing import image
from keras.models import model_from_json
# from keras.applications.inception_v3 import preprocess_input
# from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import preprocess_input


# ラベルのロード
label = open(LABEL_FILE, 'r').read().split('\n')

# モデルのロード
model = model_from_json(open(MODEL_NAME, 'r').read())
model.load_weights(sorted(Path(WEIGHT_DIR).glob('*'))[-1])

# 画像から入力データを作成
for img_path in args.filepath:
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

