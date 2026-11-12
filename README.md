# 뉴로게임 모바일

[日本語](https://github.com/hjy0221/NeuroGameMobile#%E6%97%A5%E6%9C%AC%E8%AA%9E) · [English](https://github.com/hjy0221/NeuroGameMobile#english) · [한국어](https://github.com/hjy0221/NeuroGameMobile#%ED%95%9C%EA%B5%AD%EC%96%B4)

## 日本語

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

## English

Swift + SwiftUI Canvas-based iOS 17+ on-device neural network exploration simulation v0.1.

## run

Open `NeuroGameMobile.xcodeproj` in Xcode 16 or higher and select the NeuroGameMobile scheme and the iPhone simulator to run it. On the actual device, select your Team in Signing & Capabilities. No external packages or AI API keys are required.

```sh
xcodebuild -project NeuroGameMobile.xcodeproj -scheme NeuroGameMobile -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
```

## Configuration

- Environment: movement, boundary collision, food consumption and reproduction.
- EnvironmentScenario: Separate the randomization environment used for PC training and the mobile execution evaluation environment. Training is performed in several random environments, and the app repeatedly runs the learned model in a fixed execution environment.
- SensorEncoder: 32 inputs including 16 channels of feeding direction and internal state.
- BrainSimulator: 500 artificial neurons, 4,000 sparse recursive connections, membrane potential leakage, adaptive firing threshold, and firing traces. Pure Swift calculations.
- TransformerContext: Input the last 24 frames of sensor, previous action, reward, energy, and environment tokens into Core ML Transformer to calculate the bias of the four action outputs. The current model is trained on a PC with an expert policy with bug nervous system hints.
- ActionDecoder: Calculate rotation and momentum from four sets of kinetic outputs.
- Visualization: Display the world and neural network as Canvas. Real recursive lines, firing nodes, excitatory/inhibitory colors, node selection and zoom screen.

## Model scope and extension

This is a biologically inspired artificial model and is not a reproduction of the actual Drosophila connectome. This app does not train on mobile. Train the model with multiple randomized training environments on your PC, and run the Core ML model in a fixed execution environment on your iPhone. Prey gathering performance is experimental. The simulation continues at energy 0.

The map does not change automatically within the app. A restart will also repeat the current map. Only when the user presses `새 맵` in the menu will it change to a new seed of the same execution difficulty, and the generalization behavior of the trained model and neural network activity can be observed.

By replacing the generation wiring in `BrainSimulator.buildNetwork()` with a data loader, you can introduce real connection data. `NeuralConnection` is a read-only connection representation consumed by the renderer. In the future, the world model can be added as a separate module that provides predictive input between the sensor encoder and the brain. Visualization coordinates are a fixed abstract arrangement and do not refer to anatomical locations.

`Models/NeuroContextTransformer.mlpackage` is a Core ML model for mobile execution. The currently included models were created with `Tools/train_insect_transformer.py` and trained on a PC using olfactory projection pathways, central complex direction/velocity signals, wall hazard sensors, and reward/penalty based reinforcement learning scores. In mobile apps, there is no behavioral code that directly instructs wall avoidance, and the model generates behavioral bias based on sensor input. The final training results are stored in `Models/InsectTrainingReport.json`.

A PC training data path is also available. `TrajectoryRecorder` collects recent tokens, brain output, Transformer bias, behavior, rewards, and environmental conditions during execution in `TrajectorySample` format. This JSONL sample can be put together to replace `Models/NeuroContextTransformer.mlpackage` for larger action imitation or reward-based learning. Use `Tools/create_context_transformer.py` for base model regeneration and `Tools/train_insect_transformer.py` for bug neural network hint-based training.Currently rendering and simulation is done on the main thread. When scaling to larger networks, separate simulation actors and snapshot publishing cycles.

## 한국어

Swift + SwiftUI Canvas 기반 iOS 17+ 온디바이스 신경망 탐색 시뮬레이션 v0.1.

## 실행

`NeuroGameMobile.xcodeproj`를 Xcode 16 이상에서 열고 NeuroGameMobile 스킴과 iPhone 시뮬레이터를 선택해 실행합니다. 실제 기기에서는 Signing & Capabilities에서 본인의 Team을 선택하세요. 외부 패키지나 AI API 키는 필요하지 않습니다.

```sh
xcodebuild -project NeuroGameMobile.xcodeproj -scheme NeuroGameMobile -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
```

## 구성

- Environment: 이동, 경계 충돌, 먹이 소비와 재생성.
- EnvironmentScenario: PC 훈련에 쓰는 랜덤화 환경과 모바일 실행 평가 환경을 분리. 훈련은 여러 랜덤 환경에서 수행하고, 앱은 고정 실행 환경에서 학습된 모델을 반복 실행.
- SensorEncoder: 먹이 방향 16채널과 내부 상태를 포함한 32개 입력.
- BrainSimulator: 500개 인공 뉴런, 4,000개 희소 재귀 연결, 막전위 누설, 적응 발화 문턱, 발화 흔적. 순수 Swift 계산.
- TransformerContext: 최근 24프레임의 센서, 직전 행동, 보상, 에너지, 환경 토큰을 Core ML Transformer에 입력해 네 행동 출력의 bias를 계산. 현재 모델은 PC에서 벌레 신경계 힌트를 넣은 expert 정책으로 훈련됨.
- ActionDecoder: 네 운동 출력 집단에서 회전과 추진력 계산.
- Visualization: 세계와 신경망을 Canvas로 표시. 실제 재귀 연결선, 발화 노드, 흥분성/억제성 색상, 노드 선택과 확대 화면.

## 모델 범위와 확장

생물학적 영감을 받은 인공 모델이며 실제 초파리 connectome 재현은 아닙니다. 이 앱은 모바일에서 훈련하지 않습니다. PC에서 랜덤화된 여러 훈련 환경으로 모델을 학습하고, iPhone에서는 그 Core ML 모델을 고정 실행 환경에서 실행합니다. 먹이 수집 성능은 실험적입니다. 에너지 0에서도 시뮬레이션은 계속됩니다.

앱 안에서 맵은 자동으로 바뀌지 않습니다. 다시 시작해도 현재 맵을 반복합니다. 사용자가 메뉴의 `새 맵`을 누를 때만 같은 실행 난이도의 새 seed로 바뀌며, 훈련된 모델의 일반화 행동과 신경망 활동을 관찰할 수 있습니다.

`BrainSimulator.buildNetwork()`의 생성 배선을 데이터 로더로 교체하면 실제 연결 데이터를 도입할 수 있습니다. `NeuralConnection`은 렌더러가 소비하는 읽기 전용 연결 표현입니다. 향후 world model은 센서 인코더와 뇌 사이에서 예측 입력을 제공하는 별도 모듈로 추가할 수 있습니다. 시각화 좌표는 고정된 추상 배치이며 해부학적 위치를 뜻하지 않습니다.

`Models/NeuroContextTransformer.mlpackage`는 모바일 실행용 Core ML 모델입니다. 현재 포함된 모델은 `Tools/train_insect_transformer.py`로 생성했으며, 후각 투사 경로, 중앙복합체식 방향/속도 신호, 벽 위험 센서, 보상/패널티 기반 강화학습 점수를 사용해 PC에서 훈련했습니다. 모바일 앱에는 벽 회피를 직접 지시하는 행동 코드가 없고, 모델이 센서 입력을 보고 행동 bias를 냅니다. 마지막 훈련 결과는 `Models/InsectTrainingReport.json`에 저장됩니다.

PC 훈련 데이터 경로도 준비되어 있습니다. `TrajectoryRecorder`는 실행 중 최근 토큰, 뇌 출력, Transformer bias, 행동, 보상, 환경 조건을 `TrajectorySample` 형식으로 모읍니다. 이 JSONL 샘플을 모아 더 큰 행동 모방 또는 보상 기반 학습으로 `Models/NeuroContextTransformer.mlpackage`를 교체할 수 있습니다. 기본 모델 재생성은 `Tools/create_context_transformer.py`, 벌레 신경망 힌트 기반 훈련은 `Tools/train_insect_transformer.py`를 사용합니다.

현재 렌더링과 시뮬레이션은 메인 스레드에서 수행됩니다. 더 큰 네트워크로 확장할 때는 시뮬레이션 actor와 스냅샷 게시 주기를 분리하세요.
