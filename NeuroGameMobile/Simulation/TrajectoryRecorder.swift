import Foundation

struct TrajectorySample: Encodable {
    let phase: String
    let episode: Int
    let time: Double
    let token: [Double]
    let brainOutputs: [Double]
    let transformerBias: [Double]
    let actionTurn: Double
    let actionThrust: Double
    let reward: Double
    let foodCount: Int
    let sensorRange: Double
}

final class TrajectoryRecorder {
    private(set) var sampleCount = 0
    private var pending: [TrajectorySample] = []
    private let maxPendingSamples = 1_200
    private let encoder = JSONEncoder()

    var isRecording: Bool = true
    var pendingSampleCount: Int { pending.count }

    func reset() {
        sampleCount = 0
        pending.removeAll(keepingCapacity: true)
    }

    func record(_ sample: TrajectorySample) {
        guard isRecording else { return }
        sampleCount += 1
        pending.append(sample)
        if pending.count > maxPendingSamples {
            pending.removeFirst(pending.count - maxPendingSamples)
        }
    }

    func exportPendingSamples() -> URL? {
        guard !pending.isEmpty else { return nil }
        let filename = "neurogame-trajectory-\(Int(Date().timeIntervalSince1970)).jsonl"
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(filename)
        let lines = pending.compactMap { sample -> String? in
            guard let data = try? encoder.encode(sample) else { return nil }
            return String(data: data, encoding: .utf8)
        }
        do {
            try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
            pending.removeAll(keepingCapacity: true)
            return url
        } catch {
            return nil
        }
    }
}
