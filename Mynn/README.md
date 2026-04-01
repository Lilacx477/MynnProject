# MyNN project

勉強も兼ねてニューラルネットワークをなるべく基本的な機能から実装する。

一部テストデータの準備には外部のLLMを利用している。

本筋と関係ない部分には効率的なライブラリを適宜用いる。

## Directories

```text
MYNN
│  README.md
│
├─autograd
│  │  myautograd.py
│  │  ops.py
│  └─__init__.py
│  
│
├─datasets
│  │  dataloader.py
│  │  dataset.py
│  └─__init__.py
│  
│
├─losses
│  │  lossfunctions.py
│  └─__init__.py
│
├─nn
│  │  activations.py
│  │  linear.py
│  │  mlp.py
│  │  module.py
│  └─__init__.py
│
├─optim
│  │  optimizer.py
│  └─__init__.py
│
└─tests
        mlp_test.py
```

## Requirement and Environment

* python 3.10.0
* numpy 2.2.6

## Installation

```bash
pip install numpy
```

## Usage

```bash
```

## Note

注意点などがあれば書く

## Author

作成情報を列挙する

* Lilacx477
