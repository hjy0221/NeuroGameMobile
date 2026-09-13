import CoreML
import Foundation

struct TransformerContextStatus {
    let usesCoreML: Bool
    let actionBias: [Double]

    static let empty = TransformerContextStatus(usesCoreML: false, actionBias: Array(repeating: 0, count: ActionDecoder.outputCount))
}

final class TransformerContext {
    static let contextLength = 24
    static let featureCount = 40

    private var model: MLModel?
    private var tokens = Array(
        repeating: Array(repeating: 0.0, count: TransformerContext.featureCount),
        count: TransformerContext.contextLength
    )
    private var lastAction = AgentAction(turn: 0, thrust: 0)
    private var lastReward = 0.0
    private(set) var status = TransformerContextStatus.empty
    private(set) var latestToken = Array(repeating: 0.0, count: TransformerContext.featureCount)

    init() {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        if let url = Bundle.main.url(forResource: "NeuroContextTransformer", withExtension: "mlmodelc") {
            model = try? MLModel(contentsOf: url, configuration: configuration)
        }
    }

    func reset() {
        tokens = Array(
            repeating: Array(repeating: 0.0, count: TransformerContext.featureCount),
            count: TransformerContext.contextLength
        )
        lastAction = AgentAction(turn: 0, thrust: 0)
        lastReward = 0
        latestToken = Array(repeating: 0.0, count: Self.featureCount)
        status = TransformerContextStatus.empty
    }

    func actionBias(sensors: [Double], energy: Double, phase: SimulationPhase) -> [Double] {
        appendToken(sensors: sensors, energy: energy, phase: phase)
        guard let model, let output = predict(with: model) else {
            status = TransformerContextStatus(usesCoreML: false, actionBias: fallbackBias(sensors: sensors))
            return status.actionBias
        }
        status = TransformerContextStatus(usesCoreML: true, actionBias: output)
        return output
    }

    func record(action: AgentAction, reward: Double) {
        lastAction = action
        lastReward = reward
    }

    private func appendToken(sensors: [Double], energy: Double, phase: SimulationPhase) {
        var token = Array(repeating: 0.0, count: Self.featureCount)
        for index in 0..<min(sensors.count, SensorEncoder.inputCount) {
            token[index] = sensors[index]
        }
        token[32] = lastAction.turn * 0.5 + 0.5
        token[33] = lastAction.thrust
        token[34] = min(1, max(0, lastReward / 34))
        token[35] = min(1, max(0, energy / 100))
        token[36] = phase == .training ? 1 : 0
        token[37] = sensors.prefix(16).max() ?? 0
        token[38] = sensors.prefix(16).reduce(0, +) / 16
        token[39] = 1
        tokens.removeFirst()
        tokens.append(token)
        latestToken = token
    }

    private func predict(with model: MLModel) -> [Double]? {
        guard let input = try? MLMultiArray(
            shape: [1, NSNumber(value: Self.contextLength), NSNumber(value: Self.featureCount)],
            dataType: .float32
        ) else {
            return nil
        }

        for time in 0..<Self.contextLength {
            for feature in 0..<Self.featureCount {
                input[[0, NSNumber(value: time), NSNumber(value: feature)]] = NSNumber(value: Float(tokens[time][feature]))
            }
        }

        let provider = try? MLDictionaryFeatureProvider(dictionary: ["context": MLFeatureValue(multiArray: input)])
        guard
            let provider,
            let prediction = try? model.prediction(from: provider),
            let output = prediction.featureValue(for: "actionBias")?.multiArrayValue
        else {
            return nil
        }

        return (0..<ActionDecoder.outputCount).map { index in
            let value = output[[0, NSNumber(value: index)]].doubleValue
            return min(1, max(-1, value))
        }
    }

    private func fallbackBias(sensors: [Double]) -> [Double] {
        let leftFood = sensors.prefix(8).reduce(0, +) / 8
        let rightFood = sensors.dropFirst(8).prefix(8).reduce(0, +) / 8
        let foodSignal = sensors.prefix(16).max() ?? 0
        return [
            rightFood * 0.18,
            leftFood * 0.18,
            foodSignal * 0.22,
            0
        ]
    }
}
