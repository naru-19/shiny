# pythonで作ったもの
## shiny
### 概要
- 画像一枚を入力として構成色を抽出する。(構成色数も入力する)
- 構成色を変えることで色違いのイラストを作成できる

### プログラムの説明
- 構成色の抽出にはk-meansを採用(結構実行早い)
- 色の変換には色変換リスト(col_li)の中から使うものを指定する(c=[0,10,2,3,4]のようにする)
- col_liにrgbを追加することで任意の色に変換可能(デフォルトでは15色位リストに入ってる)

### 実行例

|***input***|***output***|
|-----|-----|
|<img src="https://user-images.githubusercontent.com/61283753/127790967-71d59c16-daff-432d-a4c5-8493b2d2004c.jpg" width="200px">|<img src="https://user-images.githubusercontent.com/61283753/127791012-91998b23-cbdf-4ce6-94b6-6e7dc3c82df7.jpg" width="500px">|

### 工夫点
- 色の対応が見やすいと思う

