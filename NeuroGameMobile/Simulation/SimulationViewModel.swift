import Foundation
import SwiftUI
import QuartzCore
import Combine

@MainActor
final class SimulationViewModel: ObservableObject {
    @Published private(set) var environment: WorldEnvironment
    @Published private(set) var brainSnapshot = BrainActivitySnapshot.empty(neuronCount: 500)
    @Published private(set) var isRunning = false
    @Published private(set) var energy: Double = 80
    @Published private(set) var foodEaten = 0
    @Published private(set) var phase: SimulationPhase = .execution
    @Published private(set) var episode = 1
    @Published private(set) var scenario = EnvironmentScenario.execution
    @Published private(set) var episodeTime: Double = 0
    @Published private(set) var executionFoodEaten = 0
    @Published private(set) var transformerStatus = TransformerContextStatus.empty
    @Published private(set) var trajectorySampleCount = 0
    @Published private(set) var pendingTrajectorySampleCount = 0
    @Published private(set) var exportedTrajectoryURL: URL?
    @Published private(set) var policyActivity: Double = 0
    @Published private(set) var mapNumber = 1

    private var brain = BrainSimulator(configuration: .v01)
    private var transformer = TransformerContext()
    private var recorder = TrajectoryRecorder()
    private var sensorEncoder = SensorEncoder()
    private var actionDecoder = ActionDecoder()
    private var displayLink: CADisplayLink?
    private var lastTimestamp: CFTimeInterval?
    private var executionSeed: UInt64 = 7_303

    init() {
        let firstScenario = EnvironmentScenario.execution(seed: executionSeed)
        self.environment = WorldEnvironment(size: CGSize(width: 900, height: 620), scenario: firstScenario)
        self.scenario = firstScenario
        self.sensorEncoder.configure(for: firstScenario)
        self.brainSnapshot = brain.snapshot()
    }

    func start() {
        guard displayLink == nil else {
            isRunning = true
            return
        }
        let link = CADisplayLink(target: DisplayLinkProxy { [weak self] timestamp in
            Task { @MainActor in
                self?.tick(timestamp: timestamp)
            }
        }, selector: #selector(DisplayLinkProxy.step(_:)))
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        link.add(to: .main, forMode: .common)
        displayLink = link
        isRunning = true
    }

    func stop() {
        isRunning = false
    }

    func toggleRunning() {
        isRunning.toggle()
    }

    func reset() {
        brain = BrainSimulator(configuration: .v01)
        transformer.reset()
        recorder.reset()
        trajectorySampleCount = 0
        pendingTrajectorySampleCount = 0
        exportedTrajectoryURL = nil
        episode = 1
        executionFoodEaten = 0
        loadScenario(EnvironmentScenario.execution(seed: executionSeed))
        brainSnapshot = brain.snapshot()
        isRunning = true
    }

    func changeMap() {
        mapNumber += 1
        executionSeed = 7_303 &+ UInt64(mapNumber) &* 10_007
        reset()
    }

    private func loadScenario(_ nextScenario: EnvironmentScenario) {
        scenario = nextScenario
        sensorEncoder.configure(for: nextScenario)
        environment = WorldEnvironment(size: environment.size, scenario: nextScenario)
        transformer.reset()
        transformerStatus = transformer.status
        energy = 80
        foodEaten = 0
        episodeTime = 0
        lastTimestamp = nil
    }

    private func tick(timestamp: CFTimeInterval) {
        defer { lastTimestamp = timestamp }
        guard isRunning else { return }

        let rawDelta = lastTimestamp.map { timestamp - $0 } ?? (1.0 / 60.0)
        let dt = min(max(rawDelta, 1.0 / 120.0), 1.0 / 20.0)

        let sensors = sensorEncoder.encode(environment: environment, energy: energy)
        let outputs = brain.step(inputs: sensors, dt: dt)
        let contextBias = transformer.actionBias(sensors: sensors, energy: energy, phase: phase)
        let action = actionDecoder.decode(outputs: outputs, contextBias: contextBias)
        let result = environment.step(action: action, dt: dt)
        let reward = (result.consumedFood ? scenario.foodReward : 0) + (result.hitWall ? scenario.wallPenalty : 0)
        policyActivity = min(1, contextBias.reduce(0) { $0 + abs($1) } / Double(max(contextBias.count, 1)))
        recorder.record(TrajectorySample(
            phase: phase.rawValue,
            episode: episode,
            time: episodeTime,
            token: transformer.latestToken,
            brainOutputs: outputs,
            transformerBias: contextBias,
            actionTurn: action.turn,
            actionThrust: action.thrust,
            reward: reward,
            foodCount: scenario.foodCount,
            sensorRange: scenario.sensorRange
        ))
        trajectorySampleCount = recorder.sampleCount
        pendingTrajectorySampleCount = recorder.pendingSampleCount
        transformer.record(action: action, reward: reward)
        transformerStatus = transformer.status

        energy = min(100, max(0, energy - dt * scenario.energyDrain + max(0, reward)))
        if result.consumedFood {
            foodEaten += 1
            if phase == .execution {
                executionFoodEaten += 1
            }
        }
        if energy <= 0 {
            environment.agent.velocity *= 0.94
        }

        episodeTime += dt

        brainSnapshot = brain.snapshot()
    }

    func exportTrajectory() {
        exportedTrajectoryURL = recorder.exportPendingSamples()
        pendingTrajectorySampleCount = recorder.pendingSampleCount
    }

    deinit {
        displayLink?.invalidate()
    }
}

private final class DisplayLinkProxy: NSObject {
    private let callback: (CFTimeInterval) -> Void

    init(_ callback: @escaping (CFTimeInterval) -> Void) {
        self.callback = callback
    }

    @objc func step(_ link: CADisplayLink) {
        callback(link.timestamp)
    }
}
