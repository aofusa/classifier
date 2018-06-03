#! python3


def arg():
    """
    コマンドライン引数をパースしたオブジェクトを返します
    """
    import argparse
    parser = argparse.ArgumentParser(description='train network')
    parser.add_argument('-d', '--dataset',
                        help='path to dataset (default ./dataset.npz)',
                        default='./dataset.npz')
    parser.add_argument('-l', '--label',
                        help='label text file (default ./label.txt)',
                        default='./label.txt')
    parser.add_argument('-m', '--model',
                        help='save model json name (default ./model.json)',
                        default='./model.json')
    parser.add_argument('-w', '--weight',
                        help='save weight hdf5 name \
                            (default weights.{epoch:02d}-{val_loss:.2f}.hdf5)',
                        default='weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    parser.add_argument('-o', '--output',
                        help='save weight output directory (default ./weight)',
                        default='./weight')
    parser.add_argument('-b', '--batch',
                        help='batch size (default 128)',
                        type=int, default=128)
    parser.add_argument('-e', '--epoch',
                        help='num epochs (default 20)', type=int, default=20)
    parser.add_argument('-t', '--test_size',
                        help='num test size (default 0.1)',
                        type=float, default=0.1)
    parser.add_argument('--plaidml', action='store_true',
                       help='Enable PlaidML backend instead of Tensorflow')
    args = parser.parse_args()

    return args


def enable_plaidml():
    """
    Keras のバックエンドを tensorflow から plaidml に変更します
    """
    # Install the plaidml backend
    import plaidml.keras
    plaidml.keras.install_backend()


def load_label(filepath):
    """
    与えられた引数のファイルを改行コードで区切り配列化します
    """
    return open(filepath, 'r', encoding='utf-8').read().split('\n')


def load_dataset(filepath, label=None):
    """
    filepathで与えられたnpzファイルを読み込み
    len(label) 個のデータセットとラベルを返します
    """
    import sys
    import numpy as np
    import keras.backend as K
    from keras.utils import np_utils
    print('dataset load {}'.format(filepath), file=sys.stderr)
    dataset = np.load(filepath)
    y_dataset = np_utils.to_categorical(dataset['labels'], len(label))
    X_dataset = K.cast_to_floatx(dataset['features']) / 255.0

    return (X_dataset, y_dataset)


def split_dataset_reaction(out_x, out_y, test_size=0.1):
    """
    引数で与えられたデータセットをtest_sizeの割合で学習データと試験データに分割します
    引数で与えられたデータセットの順番は入れ替えられます
    """
    import numpy as np
    import math
    result = []
    position = 0
    for index, value in enumerate(out_y):
        if not (value == out_y[position]).all():
            result.append((position, index))
            position = index
        if index == len(out_y)-1:
            result.append((position, index+1))

    train_length = 0
    test_length = 0
    for value in result:
        length = value[1] - value[0]
        spot = math.ceil(length * (1.0 - test_size))
        train_length += spot
        test_length += (length-spot)
        out_y[train_length:len(out_y)] = np.roll(
            out_y[train_length:len(out_y)], -(length-spot), axis=0)
        out_x[train_length:len(out_x)] = np.roll(
            out_x[train_length:len(out_x)], -(length-spot), axis=0)

    X_train, X_test = out_x[0:train_length], out_x[train_length:len(out_x)]
    y_train, y_test = out_y[0:train_length], out_y[train_length:len(out_y)]

    return (X_train, X_test, y_train, y_test)


def load_model(filepath, label):
    """
    filepathで与えられたモデルがあれば読み込みます
    なかった場合、ResNet50の出力層を len(label) クラスにしたモデルを新規に作成します
    このモデルの重みの初期値は imagenet で学習済みのものとなります
    """
    import sys
    from pathlib import Path
    model = None
    if Path(filepath).exists():
        # 前回のモデルがあればロードす
        from keras.models import model_from_json
        print('model load {}'.format(filepath), file=sys.stderr)
        with open(filepath, 'r', encoding='utf-8') as f:
            model = model_from_json(f.read())

    else:
        # 学習済みモデルのロードる
        from keras.models import Model
        from keras.layers import Dense
        # from keras.layers import Input
        # from keras.applications.inception_v3 import InceptionV3
        # from keras.applications.inception_v3 import preprocess_input
        # from keras.applications.vgg19 import VGG19
        # from keras.applications.vgg19 import preprocess_input
        from keras.applications.resnet50 import ResNet50
        # from keras.applications.resnet50 import preprocess_input

        print('model not found.', file=sys.stderr)
        # pretrained_model = InceptionV3(
        #     weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
        # pretrained_model = VGG19(weights='imagenet')
        pretrained_model = ResNet50(weights='imagenet')

        # 中間層を出力するモデル
        # intermediate_layer_model = Model(
        #     inputs=pretrained_model.input,
        #     outputs=pretrained_model.layers[311].output)
        # intermediate_layer_model = Model(
        #     inputs=pretrained_model.input,
        #     outputs=pretrained_model.layers[24].output)
        intermediate_layer_model = Model(
            inputs=pretrained_model.input,
            outputs=pretrained_model.layers[175].output)

        # Denseレイヤーを接続
        x = intermediate_layer_model.output
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(len(label), activation='softmax')(x)

        # 転移学習モデル
        model = Model(
            inputs=intermediate_layer_model.input,
            outputs=predictions)

        # モデルの書き出し
        print('save model {}'.format(filepath), file=sys.stderr)
        model_json = model.to_json()
        open(filepath, 'w', encoding='utf-8').write(model_json)

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

    return model


def load_weight_reaction(out_model, filename, dirpath):
    """
    dirpathで与えられたディレクトリ配下にあるfilenameのhdf5ファイルを読み込みます
    filenameにマッチしたものがない場合、名前順にソートしたもののうち最後のものを読み込みます
    """
    import sys
    from pathlib import Path
    weights = sorted(Path(dirpath).glob('*'))
    if Path(dirpath, '.gitignore') in weights:
        weights.remove(Path(dirpath, '.gitignore'))
    if len(weights) > 0:
        w = None
        if Path(dirpath, filename).exists():
            w = Path(dirpath).joinpath(filename)
        else:
            w = weights[-1]
        print('weight load {}'.format(w), file=sys.stderr)
        out_model.load_weights(w)

    return out_model


def train_reaction(out_model, filename, dirpath,
                   x_train, y_train, x_test, y_test, epoch, batch_size):
    """
    与えられたパラメータを用いてモデルの学習を行います
    """
    import keras
    from pathlib import Path

    # 計算グラフのコンパイル
    out_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    # 途中経過の保存先の設定
    model_path = str(Path(dirpath).joinpath(filename))
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=model_path, save_best_only=True)

    # 学習の実行
    return out_model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size,
                         validation_data=(x_test, y_test),
                         callbacks=[checkpointer])


def show_evaluation(model, x, y):
    """
    引数で与えられたデータで行ったモデルの評価を出力します
    """
    loss, acc = model.evaluate(x, y)
    print('Loss {}, Accuracy {}'.format(loss, acc))

    return(loss, acc)


def main():
    # 引数のパーシング
    args = arg()

    # PlaidML backend のインストール
    if args.plaidml:
        enable_plaidml()

    # ラベルのロード
    label = load_label(args.label)

    # データセットのロード
    x, y = load_dataset(args.dataset, label)

    # データセットを学習用とテスト用に分割
    X_train, X_test, y_train, y_test = split_dataset_reaction(
        x, y, args.test_size)

    # モデルのロード
    model = load_model(args.model, label)

    # 学習済みの重みの取得
    load_weight_reaction(model, args.output, args.weight)

    # 学習の実行
    train_reaction(model, args.weight, args.output,
                   X_train, y_train, X_test, y_test, args.epoch, args.batch)

    # 評価
    show_evaluation(model, X_test, y_test)


if __name__ == '__main__':
    main()
