#! python3

ImageGenerator = None

def preprocess():
    """
    プログラム実行前の前処理
    """

    import sys
    import numpy as np
    from keras.utils import Sequence
    from progressbar import ProgressBar
    from keras.utils import np_utils
    from keras.preprocessing import image
    from keras.preprocessing.image import img_to_array, list_pictures, load_img
    from keras.applications.resnet50 import preprocess_input

    global ImageGenerator

    class internalImageGenerator(Sequence):
        """Custom image generator"""

        def __init__(self, data_paths, data_classes, 
                    batch_size=1, width=224, height=224, ch=3, num_of_class=807):
            """construction   

            :param data_paths: List of image file  
            :param data_classes: List of class  
            :param batch_size: Batch size  
            :param width: Image width  
            :param height: Image height  
            :param ch: Num of image channels  
            :param num_of_class: Num of classes  
            """

            self.data_paths = data_paths
            self.data_classes = data_classes
            self.length = len(data_paths)
            self.batch_size = batch_size
            self.width = width
            self.height = height
            self.ch = ch
            self.num_of_class = num_of_class
            self.num_batches_per_epoch = int((self.length - 1) / batch_size) + 1


        def __getitem__(self, idx):
            """Get batch data   

            :param idx: Index of batch  

            :return imgs: numpy array of images 
            :return labels: numpy array of label  
            """

            start_pos = self.batch_size * idx
            end_pos = start_pos + self.batch_size
            if end_pos > self.length:
                end_pos = self.length
            item_paths = self.data_paths[start_pos : end_pos]
            item_classes = self.data_classes[start_pos : end_pos]
            imgs = np.empty((len(item_paths), self.height, self.width, self.ch), dtype=np.float32)
            labels = np.empty((len(item_paths), self.num_of_class), dtype=int)

            error_count = 0
            prog = ProgressBar(0, end_pos - start_pos)
            for i, (item_path, item_class) in enumerate(zip(item_paths, item_classes)):
                try:
                    img = img_to_array(load_img(item_path, target_size=(self.width, self.height))) / 255.0
                    label = item_class
                    imgs[i-error_count] = img
                    labels[i-error_count] = label
                except Exception as identifier:
                    print(item_path, identifier, file=sys.stderr)
                    error_count += 1
                finally:
                    prog.update(i)
            prog.finish()

            tail_pos = end_pos - start_pos - error_count

            # imgs = preprocess_input(imgs[:tail_pos])
            # labels = np_utils.to_categorical(labels[:tail_pos], self.num_of_class)

            return imgs[:tail_pos], labels[:tail_pos]


        def __len__(self):
            """Batch length"""

            return self.num_batches_per_epoch


        def on_epoch_end(self):
            """Task when end of epoch"""
            pass

    ImageGenerator = internalImageGenerator


def arg():
    """
    コマンドライン引数をパースしたオブジェクトを返します
    """
    import argparse
    parser = argparse.ArgumentParser(description='train network')
    parser.add_argument('-d', '--dataset',
                        help='path to dataset directory (default ./dataset)',
                        default='./dataset')
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
    parser.add_argument('-s', '--img_size',
                        help='img size (default 224)',
                        type=int, default=224)
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


def make_generator(filepath, labeldata, batch_size, img_size, test_size):
    """
    画像を読み込むジェネレータを作成します
    """
    # データ一覧の取得
    import sys
    from keras.preprocessing.image import list_pictures
    from progressbar import ProgressBar
    from pathlib import Path

    # 全画像枚数
    num_pics = 0
    list_pics = []
    list_labels = []
    print('count pictures.', file=sys.stderr)
    prog = ProgressBar(0, len(labeldata))
    for index, label in enumerate(labeldata):
        path = Path(filepath).joinpath(label)
        pics = list_pictures(path)
        for p in pics:
            list_pics.append(p)
            list_labels.append(index)
        num_pics += len(pics)
        prog.update(index)
    prog.finish()
    print('{} pictures.'.format(num_pics), file=sys.stderr)

    # データジェネレータの作成
    split = int(num_pics * test_size)
    train_gen = ImageGenerator(data_paths=list_pics[split:],
                            data_classes=list_labels[split:],
                            batch_size=batch_size,
                            width=img_size,
                            height=img_size,
                            num_of_class=len(labeldata))
    val_gen = ImageGenerator(data_paths=list_pics[:split],
                            data_classes=list_labels[:split],
                            batch_size=batch_size,
                            width=img_size,
                            height=img_size,
                            num_of_class=len(labeldata))

    return (train_gen, val_gen)


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


def train_generator_reaction(out_model, filename, dirpath,
                   train_gen, val_gen, epoch):
    """
    与えられたパラメータを用いてモデルの学習を行います
    """
    import keras
    import numpy as np
    from pathlib import Path
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import np_utils
    from keras.preprocessing import image
    from keras.preprocessing.image import img_to_array, list_pictures, load_img

    # 計算グラフのコンパイル
    out_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    # 途中経過の保存先の設定
    model_path = str(Path(dirpath).joinpath(filename))
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=model_path, save_best_only=True)

    # 学習の実行
    return out_model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.num_batches_per_epoch,
        validation_data=val_gen,
        validation_steps=val_gen.num_batches_per_epoch,
        epochs=epoch,
        shuffle=True,
        callbacks=[checkpointer])


def show_evaluation(model, x, y):
    """
    引数で与えられたデータで行ったモデルの評価を出力します
    """
    loss, acc = model.evaluate_generator(x)
    print('Loss {}, Accuracy {}'.format(loss, acc))

    return(loss, acc)


def main():
    # 引数のパーシング
    args = arg()

    # PlaidML backend のインストール
    if args.plaidml:
        enable_plaidml()

    # プログラム実行前処理
    preprocess()

    # ラベルのロード
    label = load_label(args.label)

    # データジェネレータの取得
    (train_gen, val_gen) = make_generator(args.dataset, label, args.batch, args.img_size, args.test_size)

    # モデルのロード
    model = load_model(args.model, label)

    # 学習済みの重みの取得
    load_weight_reaction(model, args.output, args.weight)

    # 学習の実行
    train_generator_reaction(model, args.weight, args.output,
                   train_gen, val_gen, args.epoch)

    # モデルの保存
    from pathlib import Path
    model.save_weights(Path(args.output).joinpath(args.weight))

    # 評価
    show_evaluation(model, train_gen, val_gen)


if __name__ == '__main__':
    main()
