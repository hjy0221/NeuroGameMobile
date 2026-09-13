# 뉴로게임 모바일

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
