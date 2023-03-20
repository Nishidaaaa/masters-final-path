1. gitからclone
2. Docker build
    make docker-build
3. Docker run
    make docker-run

---以下コンテナの中
4. mlflowのUIサーバー立ち上げ
    nohup mlflow ui --port 5000 > /dev/null &



5. 訓練実行
    python run_all.py 