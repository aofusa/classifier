#! python3


def arg():
    """
    コマンドライン引数をパースしたオブジェクトを返します
    """
    import argparse
    parser = argparse.ArgumentParser(description='make dataset')
    parser.add_argument('-d', '--dir',
                        help='path to dataset directory (default ./dataset)',
                        default='./dataset')
    parser.add_argument('-l', '--label',
                        help='label text file (default ./label.txt)',
                        default='./label.txt')
    parser.add_argument('-o', '--output',
                        help='save numpy file (.npz) name \
                             (default ./dataset.npz)',
                        default='./dataset.npz')
    parser.add_argument('-s', '--size',
                        help='resize image (default 224)',
                        type=int, default=224)
    args = parser.parse_args()

    return args


def load_label(filepath):
    """
    与えられた引数のファイルを改行コードで区切り配列化します
    """
    return open(filepath, 'r', encoding='utf-8').read().split('\n')


def count_pictures(labeldata, datapath):
    """
    datapathで与えられたディレクトリ内に存在するlabeldataで指定されたディレクトリ以下の画像の枚数を取得します
    """
    from keras.preprocessing.image import list_pictures
    import sys
    from pathlib import Path
    from progressbar import ProgressBar
    print('count pictures.', file=sys.stderr)
    result = 0
    prog = ProgressBar(0, len(labeldata))
    for index, label in enumerate(labeldata):
        path = Path(datapath).joinpath(label)
        pics = list_pictures(path)
        result += len(pics)
        prog.update(index+1)
    prog.finish()
    print('{} pictures.'.format(result), file=sys.stderr)

    return result


def allocate_memory(count, size):
    """
    count*size^2*3 + count*4 byteのメモリを確保します
    """
    import sys
    import numpy as np
    print('reserve memory {} byte.'.format(
        count*size*size*3*1 + count*4), file=sys.stderr)
    Y = np.empty(count, dtype=int)
    X = np.empty((count, size, size, 3), dtype='int8')

    return (X, Y)


def make_dataset_reaction(out_x, out_y, labeldata, datapath, size):
    """
    datapathで指定されたディレクトリ内のllabeldataで指定されたディレクトリ以下の画像ファイルをsize*sizeにリサイズしout_x・out_yに格納します
    また、読み込みに失敗した画像パスのリストを返します
    """
    from keras.preprocessing.image import img_to_array, list_pictures, load_img
    import sys
    import numpy as np
    from pathlib import Path
    from progressbar import ProgressBar
    IMG = np.empty((1, size, size, 3), dtype='int8')
    POSITION = 0
    ERROR = []
    for index, label in enumerate(labeldata):
        path = Path(datapath).joinpath(label)
        pics = list_pictures(path)
        prog = ProgressBar(0, len(pics))
        print('[{}/{}] load {} {} pictures.'.format(
            index+1, len(labeldata), path, len(pics)), file=sys.stderr)
        count = 0
        for picture in pics:
            try:
                IMG[0] = img_to_array(
                    load_img(picture, target_size=(size, size)))
                out_x[POSITION+count] = IMG
                prog.update(count+1)
                count += 1
            except Exception as identifier:
                print(picture, identifier)
                pics.remove(picture)
                ERROR.append(picture)
        out_y[POSITION:POSITION+len(pics)] = np.full(
            len(pics), index, dtype=int)
        POSITION += len(pics)
        prog.finish()

    return ERROR


def save_dataset(filepath, x, y, end=0):
    """
    filepathにx・yを-endまでの範囲でファイルに保存します
    """
    import numpy as np
    if end > 0:
        np.savez_compressed(filepath, features=x[:-end], labels=y[:-end])
    else:
        np.savez_compressed(filepath, features=x, labels=y)


def main():
    # 引数のパーシング
    args = arg()

    # ラベルの読み込み
    label = load_label(args.label)

    # 全画像枚数の取得
    count = count_pictures(label, args.dir)

    # 画像とラベルデータのメモリを確保
    (x, y) = allocate_memory(count, args.size)

    # 対象画像の読み込みとラベリング
    error = make_dataset_reaction(x, y, label, args.dir, args.size)

    # データセットを保存
    save_dataset(args.output, x, y, len(error))


if __name__ == '__main__':
    main()
