# 뉴로게임 모바일

[日本語](README.ja.md) · [English](README.md) · [한국어](README.ko.md)

Swift + SwiftUI CanvasベースのiOS 17+オンデバイスニューラルネットワークナビゲーションシミュレーションv0.1。

## 実行

`NeuroGameMobile.xcodeproj`をXcode 16以降で開き、NeuroGameMobileスキームとiPhoneシミュレータを選択して実行します。実際の端末では、Signing & Capabilitiesで自分のチームを選択してください。外部パッケージやAI APIキーは必要ありません。

```sh
xcodebuild -project NeuroGameMobile.xcodeproj -scheme NeuroGameMobile -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
```

## 設定

- Environment: 移動、境界の衝突、餌の消費と再生成。
- EnvironmentScenario：PCトレーニングに使うランダム化環境とモバイル実行評価環境を分離。トレーニングは複数のランダム環境で行われ、アプリは固定実行環境で学習されたモデルを繰り返し実行。
- SensorEncoder：供給方向16チャンネルと内部状態を含む32個の入力。
- BrainSimulator：500個の人工ニューロン、4,000個の希少再帰接続、膜電位漏洩、適応発火閾値、発火跡。純粋なSwiftの計算。
- TransformerContext:最近24フレームのセンサー、直前行動、補償、エネルギー、環境トークンをCore ML Transformerに入力して4行動出力のbiasを計算。現在のモデルは、PCでワーム神経系ヒントを入れたexpertポリシーで訓練されている。
- ActionDecoder：4つの運動出力集団での回転と推進力の計算。
- ビジュアル化：世界とニューラルネットワークをキャンバスとしてマーク。実際の再帰接続線、発話ノード、興奮性/抑制性色、ノード選択と拡大画面。

## モデルの範囲と拡張

生物学的インスピレーションを受けた人工モデルであり、実際のチョウハエコネクトム再現ではありません。このアプリはモバイルで訓練されていません。 PCでランダム化された複数のトレーニング環境でモデルを学習し、iPhoneではそのCore MLモデルを固定実行環境で実行します。エサ収集性能は実験的です。エネルギー０でもシミュレーションは継続する。

アプリ内では、マップは自動的に変わりません。再起動しても現在のマップを繰り返します。ユーザーがメニューの`새 맵`を押したときにのみ、同じ実行難易度の新しいシードに変わり、トレーニングされたモデルの一般化行動とニューラルネットワーク活動を観察できます。

`BrainSimulator.buildNetwork()`の生成配線をデータローダに置き換えることで、実際の接続データを導入できます。 `NeuralConnection`は、レンダラーが消費する読み取り専用接続表現です。将来の世界モデルは、センサーエンコーダーと脳の間に予測入力を提供する別々のモジュールとして追加することができます。可視化座標は固定された抽象配置であり、解剖学的位置を意味しません。

`Models/NeuroContextTransformer.mlpackage`はモバイル実行用のCore MLモデルです。現在含まれているモデルは`Tools/train_insect_transformer.py`で作成され、嗅覚投影経路、中央複合体方向/速度信号、壁危険センサー、補償/パネルベースの強化学習スコアを使用してPCで訓練されました。モバイルアプリには壁回避を直接指示する行動コードがなく、モデルがセンサー入力を見て行動biasを出します。最後のトレーニング結果は`Models/InsectTrainingReport.json`に保存されます。

PCトレーニングデータパスも用意されています。 `TrajectoryRecorder`は、実行中の最近のトークン、脳出力、Transformer bias、行動、報酬、環境条件を`TrajectorySample`形式にまとめます。このJSONLサンプルを集めて、より大きな行動模倣または補償ベースの学習で`Models/NeuroContextTransformer.mlpackage`を置き換えることができます。基本モデルの再生は`Tools/create_context_transformer.py`、ワームニューラルネットワークのヒントベースのトレーニングは`Tools/train_insect_transformer.py`を使用しています。現在のレンダリングとシミュレーションはメインスレッドで行われます。より大きなネットワークに拡張する場合は、シミュレーションactorとスナップショット公開サイクルを分離してください。
