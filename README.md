classifier
---


好きな画像とCNNを使ったクラス分類機の学習セット  


### 環境構築
```
pip3 install -r requirements.txt
```


### 学習画像の用意
以下のような感じで各ファイルが配置されていること  
データセットフォルダ  
- dataset
```
dataset
├── Eevee
│   ├── Picture1.jpg
│   └── Picture2.png
├── Jolteon
│   ├── Picture1.jpg
│   └── Picture2.png
├── Vaporeon
│   ├── Picture1.jpg
│   └── Picture2.png
└── Flareon
    ├── Picture1.jpg
    └── Picture2.png
```

ラベルファイル  
- label.txt  
```
Eevee
Jolteon
Vaporeon
Flareon
```

学習用データセットの生成  
```
python3 make_data.py --dir ./dataset --label ./label.txt --output ./dataset.npz
```


### 学習
```
python3 train_network.py --dataset ./dataset.npz --label ./label.txt --model ./model.json --weight weight.hdf5 --output ./weight --batch 128 --epoch 20
```


### 推論
```
python3 predict.py --label ./label.txt --model ./model.json --weight ./weight/weight.hdf5 <画像ファイルパス>
```

以下のようにすると各データセットの正答率が見えて楽しい  
```
find ./dataset/* -maxdepth 0 | while read -r val; do find "$val" -type f | head -n 1; done | sed 's/\n/ /g' | xargs python3 predict.py --label ./label.txt --model ./model.json --weight $(find ./weight | tail -1)
```


### 推論(NNVM)
別途NNVMとTVM環境の用意が必要  
```
python3 predict_nnvm.py --label ./label.txt --model ./model.json --weight ./weight --opencl <画像ファイルパス>
```


### その他
crawler ディレクトリ配下に使用した画像収集スクリプトを置いておきます  
make_data.py 実行時、使用するデータセットの大きさに応じて相応のスワップ領域（仮想メモリ）を用意しないと Out of Memory で落ちます  
Windows だと文字コードを UTF-8 にしないとダメかもしれない  
- chcp 65001  
  Powershell の文字コードを UTF-8 に変更  
  (chcp 932 で元に戻る、あらかじめ chcp コマンドで元の文字コードを確認することを推奨)  

