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
```
dataset
├── Eevee
    ├── Picture1.jpg
    └── Picture2.png
├── Jolteon
    ├── Picture1.jpg
    └── Picture2.png
├── Vaporeon
    ├── Picture1.jpg
    └── Picture2.png
└── Flareon
    ├── Picture1.jpg
    └── Picture2.png
```

ラベルファイル  
```
Eevee
Jolteon
Vaporeon
Flareon
```

学習用データセットの生成  
```
python3 make_data.py
```


### 学習
```
python3 train_network.py
```


### 推論
```
python3 predict.py <画像ファイルパス>
```


### 推論(NNVM)
別途NNVMとTVM環境の用意が必要  
```
python3 predict_nnvm.py <画像ファイルパス>
```


### その他
crawler ディレクトリ配下に使用した画像収集スクリプトを置いておく  

