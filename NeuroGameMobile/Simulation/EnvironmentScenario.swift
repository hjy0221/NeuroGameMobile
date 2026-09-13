import Foundation

enum SimulationPhase: String, CaseIterable, Identifiable {
    case training
    case execution

    var id: String { rawValue }

    var title: String {
        switch self {
        case .training: "학습"
        case .execution: "실행"
        }
    }

    var subtitle: String {
        switch self {
        case .training: "훈련 환경"
        case .execution: "실행 환경"
        }
    }
}

struct EnvironmentScenario {
    let seed: UInt64
    let foodCount: Int
    let foodRadius: Double
    let turnRate: Double
    let acceleration: Double
    let damping: Double
    let maxSpeed: Double
    let sensorRange: Double
    let sensorNoise: Double
    let energyDrain: Double
    let foodReward: Double
    let wallPenalty: Double

    static let execution = EnvironmentScenario(
        seed: 7_303,
        foodCount: 24,
        foodRadius: 8.5,
        turnRate: 5.9,
        acceleration: 420,
        damping: 0.945,
        maxSpeed: 330,
        sensorRange: 260,
        sensorNoise: 0,
        energyDrain: 2.0,
        foodReward: 26,
        wallPenalty: -14
    )

    static func execution(seed: UInt64) -> EnvironmentScenario {
        EnvironmentScenario(
            seed: seed,
            foodCount: 24,
            foodRadius: 8.5,
            turnRate: 5.9,
            acceleration: 420,
            damping: 0.945,
            maxSpeed: 330,
            sensorRange: 260,
            sensorNoise: 0,
            energyDrain: 2.0,
            foodReward: 26,
            wallPenalty: -14
        )
    }

    static func training(episode: Int, seed: UInt64) -> EnvironmentScenario {
        var random = SeededRandom(seed: seed &+ UInt64(episode) &* 1_009)
        return EnvironmentScenario(
            seed: seed &+ UInt64(episode) &* 97,
            foodCount: random.nextInt(18...28),
            foodRadius: random.nextDouble(in: 7.5...11.5),
            turnRate: random.nextDouble(in: 5.0...7.0),
            acceleration: random.nextDouble(in: 360...500),
            damping: random.nextDouble(in: 0.925...0.97),
            maxSpeed: random.nextDouble(in: 280...380),
            sensorRange: random.nextDouble(in: 210...320),
            sensorNoise: random.nextDouble(in: 0.015...0.08),
            energyDrain: random.nextDouble(in: 1.6...3.1),
            foodReward: random.nextDouble(in: 18...34),
            wallPenalty: random.nextDouble(in: -18 ... -8)
        )
    }
}
