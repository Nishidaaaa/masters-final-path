# 概要
修士論文で制作した自己教師あり学習(DINO)とkMeans,RFを組み合わせた画像分類器です．
高解像度な画像を分割し自己教師を適用することで，計算資源上の問題をクリアし，また，判断要因となる画像領域の考察を行うことができます．

mlflowで実験データを管理しています．
mlflowはexperiment

# 環境構築
1. データの配置
    - `git clone このリポジトリ`
    - 画像データを`./dataset/`以下に配置
2. Docker build

    `make docker-build`

3. Docker run


    `make docker-run`

---以下コンテナの中---


4. mlflowのUIサーバー立ち上げ


    `nohup mlflow ui --port 5000 > /dev/null & `


5. 訓練実行

    `python run_all.py`

6. 実験結果確認

    [localhost:5000](http://localhost:5000)

