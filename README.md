## ReadMe
このプログラムはデープラーニングでゲームの攻略をします。

### 環境とか
cythonをインストールしてね

### 導入とか
- ソースコードを持ってくる 
- pythonのコマンドラインで次のコマンドを実行 
    python setup.py build_ext -i 
- train2.pyでトレーディング開始 
    train2.pyを書き換えて設定の変更をしてください。
    play_batch_size = 全体で何ループのトレーニングをするか 
    play_parallel_size = 1ループで何個の試合をするか（並列に行われます） 
    check_freq = 何ループごとに対戦を確認するか 
- tcp_play.py => tcpでプレーします 
