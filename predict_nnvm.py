#! python3


def arg():
    """
    コマンドライン引数をパースしたオブジェクトを返します
    """
    import argparse
    parser = argparse.ArgumentParser(description='predict image \
        on target device api with NNVM/TVM')
    parser.add_argument('filepath', help='image file path', nargs='+')
    parser.add_argument('-l', '--label',
                        help='label text file (default ./label.txt)',
                        default='./label.txt')
    parser.add_argument('-m', '--model',
                        help='load model json (default ./model.json)',
                        default='./model.json')
    parser.add_argument('-w', '--weight',
                        help='load weight hdf5 file \
                            (default ./weight/weight.hdf5)',
                        default='./weight/weight.hdf5')
    parser.add_argument('-d', '--display',
                        help='number of display result (default 5)',
                        type=int, default=5)
    parser.add_argument('-c', '--context',
                        help='target running context (default 0)',
                        type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--opencl', action='store_true',
                       help='Target API OpenCL (default)')
    group.add_argument('--llvm', action='store_true',
                       help='Target API LLVM CPU')
    group.add_argument('--cuda', action='store_true',
                       help='Target API CUDA')
    group.add_argument('--metal', action='store_true',
                       help='Target API Metal')
    group.add_argument('--opengl', action='store_true',
                       help='Target API OpenGL')
    group.add_argument('--vulkan', action='store_true',
                       help='Target API Vulkan')
    args = parser.parse_args()

    TARGET_LIST = ['opencl', 'llvm', 'cuda', 'metal', 'opengl', 'vulkan']
    args.api = TARGET_LIST[0]

    if args.opencl:
        args.api = TARGET_LIST[0]
    if args.llvm:
        args.api = TARGET_LIST[1]
    if args.cuda:
        args.api = TARGET_LIST[2]
    if args.metal:
        args.api = TARGET_LIST[3]
    if args.opengl:
        args.api = TARGET_LIST[4]
    if args.vulkan:
        args.api = TARGET_LIST[5]

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


def show_predict(model, label, filelist, display=5, api='opencl', context=0):
    """
    引き数で与えられたパスの画像を読み込み推論結果を出力します
    推論結果はjson形式で出力されます
    また、contextで指定されたプラットフォーム上で
    apiで指定されたプラットフォーム用にコンパイルし実行されます
    """
    import nnvm
    import tvm
    from tvm.contrib import graph_runtime
    import numpy as np
    import keras
    from keras.preprocessing import image
    # from keras.applications.inception_v3 import preprocess_input
    # from keras.applications.vgg19 import preprocess_input
    from keras.applications.resnet50 import preprocess_input

    # Kerasのモデルをtvmでコンパイルする
    # from keras.models import load_model
    # model = load_model('model.h5')
    # model.compile(loss='categorical_crossentropy',
    #             optimizer='adam',
    #             metrics=['accuracy'])

    # KerasのモデルをNNVMで読み込む
    sym, params = nnvm.frontend.from_keras(model)

    # CoreML を経由して読み込む (Only Python2)
    # import coremltools
    # model.save('model.h5')
    # coreml_model = coremltools.converters.keras.convert('model.h5',
    #                                         input_names='features',
    #                                         image_input_names='labels',
    #                                         class_labels='label.txt')
    # sym, params = nnvm.frontend.from_coreml(coreml_model)

    # モデルのコンパイル
    shape_dict = {'data': (1, 3, 224, 224)}
    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(
            sym, api, shape_dict, params=params)

    # モデルを指定されたコンテキスト上にロードする
    ctx = tvm.context(api, context)
    m = graph_runtime.create(graph, lib, ctx)

    # JSON形式で表示
    print('[')
    for img_path in filelist:
        # 入力データのセット
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x).transpose([0, 3, 1, 2])

        # 推論実行
        m.set_input('data', tvm.nd.array(x.astype(keras.backend.floatx())))
        m.set_input(**params)
        m.run()

        # 計算結果を取得し、推論結果を確認
        out_shape = (len(label),)
        tvm_out = m.get_output(0, tvm.nd.empty(
            out_shape, keras.backend.floatx())).asnumpy()
        pred = dict(zip(label, tvm_out))
        ranking = sorted(pred.items(), key=lambda x: -x[1])

        # 推論結果を出力
        print('  {')
        print('    "File": "{}",'.format(img_path))
        print('    "Predict": "{}",'.format(label[np.argmax(tvm_out)]))
        print('    "Ranking": {')
        for i, v in enumerate(ranking):
            if i >= display-1 or i == len(ranking)-1:
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
    show_predict(model, label, args.filepath, args.display,
                 args.api, args.context)


if __name__ == '__main__':
    main()
