#! python3


def arg():
    """
    コマンドライン引数をパースしたオブジェクトを返します
    """
    import argparse
    parser = argparse.ArgumentParser(description='predict image')
    parser.add_argument('filepath', help='image file path', nargs='+')
    parser.add_argument('-l', '--label',
                        help='label text file (default ./label.txt)',
                        default='./label.txt')
    parser.add_argument('-m', '--model',
                        help='load model json (default ./model.json)',
                        default='./model.json')
    parser.add_argument('-w', '--weight',
                        help='load hdf5 weight file \
                            (default ./weight/weight.hdf5)',
                        default='./weight/weight.hdf5')
    parser.add_argument('-d', '--display',
                        help='number of display result (default 5)',
                        type=int, default=5)
    args = parser.parse_args()

    return args


def load_label(filepath):
    """
    与えられた引数のファイルを改行コードで区切り配列化します
    """
    return open(filepath, 'r', encoding='utf-8').read().split('\n')


def load_model(modelpath, weightpath):
    """
    引数で与えられたモデルと重みを読み込みます
    """
    from keras.models import model_from_json
    model = model_from_json(open(modelpath, 'r', encoding='utf-8').read())
    model.load_weights(weightpath)

    return model


def show_predict(model, label, filelist, display=5):
    """
    引き数で与えられたパスの画像を読み込み推論結果を出力します
    推論結果はjson形式で出力されます
    """
    import numpy as np
    from keras.preprocessing import image
    # from keras.applications.inception_v3 import preprocess_input
    # from keras.applications.vgg19 import preprocess_input
    from keras.applications.resnet50 import preprocess_input
    print('[')
    for img_path in filelist:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # 推論実行
        preds = model.predict(x)
        pred = dict(zip(label, preds[0]))
        ranking = sorted(pred.items(), key=lambda x: -x[1])

        # 推論結果を出力
        print('  {')
        print('    "File": "{}",'.format(img_path))
        print('    "Predict": "{}",'.format(label[np.argmax(preds)]))
        print('    "Ranking": {')
        for i, v in enumerate(ranking):
            if i == len(ranking)-1 or (display>0 and i >= display-1):
                print('      "{}": {} "Label":"{}", "Accuracy":{} {}'.format(
                    i, '{', v[0], v[1], '}'))
                break
            else:
                print('      "{}": {} "Label":"{}", "Accuracy":{} {},'.format(
                    i, '{', v[0], v[1], '}'))
        print('    }')
        if img_path != filelist[-1]:
            print('  },')
        else:
            print('  }')

    print(']')


def main():
    # 引数のパーシング
    args = arg()

    # ラベルのロード
    label = load_label(args.label)

    # モデルのロード
    model = load_model(args.model, args.weight)

    # 画像から推論を実行して結果を表示
    show_predict(model, label, args.filepath, args.display)


if __name__ == '__main__':
    main()
