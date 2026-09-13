import Foundation

struct AgentAction {
    var turn: Double
    var thrust: Double
}

struct ActionDecoder {
    static let outputCount = 4

    func decode(outputs: [Double], contextBias: [Double] = []) -> AgentAction {
        let left = biased(outputs, contextBias, 0)
        let right = biased(outputs, contextBias, 1)
        let forward = biased(outputs, contextBias, 2)
        let brake = biased(outputs, contextBias, 3)

        let turn = max(-1, min(1, right - left))
        let thrust = max(0, min(1, 0.14 + forward - brake * 0.45))
        return AgentAction(turn: turn, thrust: thrust)
    }

    private func biased(_ outputs: [Double], _ contextBias: [Double], _ index: Int) -> Double {
        let output = outputs[safe: index] ?? 0
        let bias = contextBias[safe: index] ?? 0
        return min(1, max(0, output + bias * 0.42))
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
