# 뉴로게임 모바일

[日本語](README.ja.md) · [English](README.md) · [한국어](README.ko.md)

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
