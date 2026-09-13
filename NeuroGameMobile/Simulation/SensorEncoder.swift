import Foundation

struct SensorEncoder {
    static let inputCount = 32

    var range: Double = 240
    var rayCount: Int = 16
    var noise: Double = 0
    private var random = SeededRandom(seed: 4_404)

    mutating func configure(for scenario: EnvironmentScenario) {
        range = scenario.sensorRange
        noise = scenario.sensorNoise
    }

    mutating func encode(environment: WorldEnvironment, energy: Double) -> [Double] {
        var inputs = Array(repeating: 0.0, count: Self.inputCount)
        let agent = environment.agent

        for food in environment.foods {
            let dx = Double(food.position.x - agent.position.x)
            let dy = Double(food.position.y - agent.position.y)
            let distance = hypot(dx, dy)
            guard distance <= range else { continue }

            let relativeAngle = normalizeAngle(atan2(dy, dx) - agent.heading)
            let normalized = (relativeAngle + .pi) / (2 * .pi)
            let bin = min(rayCount - 1, max(0, Int(normalized * Double(rayCount))))
            let signal = 1.0 - distance / range
            inputs[bin] = max(inputs[bin], signal)
        }

        let speed = hypot(agent.velocity.dx, agent.velocity.dy)
        inputs[16] = min(1, speed / environment.scenario.maxSpeed)
        inputs[17] = (energy / 100)
        inputs[18] = sin(agent.heading) * 0.5 + 0.5
        inputs[19] = cos(agent.heading) * 0.5 + 0.5
        inputs[20] = Double(agent.position.x / environment.size.width)
        inputs[21] = Double(agent.position.y / environment.size.height)
        inputs[22] = wallRisk(environment: environment, angle: agent.heading)
        inputs[23] = wallRisk(environment: environment, angle: agent.heading - .pi * 0.5)
        inputs[24] = wallRisk(environment: environment, angle: agent.heading + .pi * 0.5)
        inputs[25] = edgeRisk(environment: environment)

        for index in 26..<Self.inputCount {
            let phase = Double(index) * 12.9898 + Double(environment.foods.count) * 78.233
            inputs[index] = (sin(phase) + 1) * 0.05
        }

        guard noise > 0 else { return inputs }
        return inputs.map { value in
            min(1, max(0, value + random.nextDouble(in: (-noise)...noise)))
        }
    }

    private func normalizeAngle(_ angle: Double) -> Double {
        var value = angle
        while value > .pi { value -= 2 * .pi }
        while value < -.pi { value += 2 * .pi }
        return value
    }

    private func wallRisk(environment: WorldEnvironment, angle: Double) -> Double {
        let origin = environment.agent.position
        let dx = cos(angle)
        let dy = sin(angle)
        let width = Double(environment.size.width)
        let height = Double(environment.size.height)
        var candidates: [Double] = []
        if dx > 0 {
            candidates.append((width - Double(origin.x)) / dx)
        } else if dx < 0 {
            candidates.append((0 - Double(origin.x)) / dx)
        }
        if dy > 0 {
            candidates.append((height - Double(origin.y)) / dy)
        } else if dy < 0 {
            candidates.append((0 - Double(origin.y)) / dy)
        }
        let distance = candidates.filter { $0 > 0 }.min() ?? range
        return min(1, max(0, 1 - distance / 130))
    }

    private func edgeRisk(environment: WorldEnvironment) -> Double {
        let agent = environment.agent
        let distance = min(
            Double(agent.position.x),
            Double(agent.position.y),
            Double(environment.size.width - agent.position.x),
            Double(environment.size.height - agent.position.y)
        )
        return min(1, max(0, 1 - distance / 100))
    }
}
