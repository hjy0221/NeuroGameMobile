import CoreGraphics
import Foundation

struct WorldEnvironment {
    struct Agent {
        var position: CGPoint
        var heading: Double
        var velocity: CGVector
        var radius: CGFloat = 12
    }

    struct Food: Identifiable {
        let id: Int
        var position: CGPoint
        var radius: CGFloat
    }

    struct StepResult {
        let consumedFood: Bool
        let hitWall: Bool
    }

    let size: CGSize
    let scenario: EnvironmentScenario
    var agent: Agent
    var foods: [Food]
    private var random: SeededRandom

    init(size: CGSize, scenario: EnvironmentScenario) {
        self.size = size
        self.scenario = scenario
        self.random = SeededRandom(seed: scenario.seed)
        self.agent = Agent(
            position: CGPoint(x: size.width * 0.5, y: size.height * 0.5),
            heading: random.nextDouble(in: (-Double.pi)...Double.pi),
            velocity: .zero
        )
        self.foods = []
        for id in 0..<scenario.foodCount {
            foods.append(Food(
                id: id,
                position: WorldEnvironment.randomPoint(in: size, random: &random),
                radius: CGFloat(scenario.foodRadius)
            ))
        }
    }

    mutating func step(action: AgentAction, dt: Double) -> StepResult {
        agent.heading += action.turn * dt * scenario.turnRate
        let direction = CGVector(dx: cos(agent.heading), dy: sin(agent.heading))
        agent.velocity.dx += direction.dx * action.thrust * dt * scenario.acceleration
        agent.velocity.dy += direction.dy * action.thrust * dt * scenario.acceleration
        agent.velocity *= scenario.damping

        let speed = hypot(agent.velocity.dx, agent.velocity.dy)
        if speed > scenario.maxSpeed {
            agent.velocity *= scenario.maxSpeed / speed
        }

        agent.position.x += agent.velocity.dx * dt
        agent.position.y += agent.velocity.dy * dt
        let hitWall = keepAgentInBounds()

        var consumed = false
        for index in foods.indices {
            let dx = foods[index].position.x - agent.position.x
            let dy = foods[index].position.y - agent.position.y
            let distance = hypot(dx, dy)
            if distance < foods[index].radius + agent.radius {
                foods[index].position = Self.randomPoint(in: size, random: &random)
                foods[index].radius = CGFloat(scenario.foodRadius)
                consumed = true
            }
        }
        return StepResult(consumedFood: consumed, hitWall: hitWall)
    }

    private mutating func keepAgentInBounds() -> Bool {
        let margin = agent.radius
        var hitWall = false
        if agent.position.x < margin {
            agent.position.x = margin
            hitWall = true
        } else if agent.position.x > size.width - margin {
            agent.position.x = size.width - margin
            hitWall = true
        }

        if agent.position.y < margin {
            agent.position.y = margin
            hitWall = true
        } else if agent.position.y > size.height - margin {
            agent.position.y = size.height - margin
            hitWall = true
        }
        if hitWall {
            agent.velocity = .zero
        }
        return hitWall
    }

    private static func randomPoint(in size: CGSize, random: inout SeededRandom) -> CGPoint {
        CGPoint(
            x: CGFloat(random.nextDouble(in: 32...(Double(size.width) - 32))),
            y: CGFloat(random.nextDouble(in: 32...(Double(size.height) - 32)))
        )
    }
}

extension CGVector {
    static func *= (lhs: inout CGVector, rhs: Double) {
        lhs.dx *= rhs
        lhs.dy *= rhs
    }
}
