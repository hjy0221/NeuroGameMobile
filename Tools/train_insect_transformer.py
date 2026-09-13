from pathlib import Path
import json
import math
import shutil

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

CONTEXT_LENGTH = 24
FEATURE_COUNT = 40
EMBED_DIM = 32
OUTPUT_COUNT = 4
SAMPLE_COUNT = 72_000


def make_projection(seed=20260913):
    rng = np.random.default_rng(seed)
    wq = rng.normal(0, 0.045, (FEATURE_COUNT, EMBED_DIM)).astype(np.float32)
    wk = rng.normal(0, 0.045, (FEATURE_COUNT, EMBED_DIM)).astype(np.float32)
    wv = rng.normal(0, 0.035, (FEATURE_COUNT, EMBED_DIM)).astype(np.float32)

    # Olfactory projection neurons: food direction ring -> compact directional code.
    for food_bin in range(16):
        angle = (food_bin + 0.5) / 16 * 2 * math.pi - math.pi
        wv[food_bin, 0] += math.cos(angle) * 1.25
        wv[food_bin, 1] += math.sin(angle) * 1.25
        wv[food_bin, 2 + food_bin % 6] += 0.72
        wq[food_bin, 0] += math.cos(angle) * 0.36
        wk[food_bin, 1] += math.sin(angle) * 0.36

    # Central-complex-like heading and speed channels.
    wv[16, 8] = 0.9
    wv[18, 9] = 0.75
    wv[19, 10] = 0.75
    wv[20, 11] = 0.65
    wv[21, 12] = 0.65

    # Touch/edge reflex channels.
    for feature, embed in [(22, 13), (23, 14), (24, 15), (25, 16)]:
        wv[feature, embed] = 1.25
        wq[feature, embed] = 0.5
        wk[feature, embed] = 0.5

    # Mushroom-body-like reward and previous-action memory.
    for feature, embed in [(32, 17), (33, 18), (34, 19), (35, 20), (36, 21), (37, 22), (38, 23)]:
        wv[feature, embed] = 0.85
        wq[feature, embed] = 0.3
        wk[feature, embed] = 0.3

    return wq.astype(np.float32), wk.astype(np.float32), wv.astype(np.float32)


def expert_action(token):
    food = token[:16]
    food_strength = float(food.max())
    if food_strength > 0:
        angles = np.linspace(-math.pi + math.pi / 16, math.pi - math.pi / 16, 16)
        x = float(np.sum(food * np.cos(angles)))
        y = float(np.sum(food * np.sin(angles)))
        food_angle = math.atan2(y, x)
    else:
        food_angle = 0.0

    forward_wall = float(token[22])
    left_wall = float(token[23])
    right_wall = float(token[24])
    edge = float(token[25])
    previous_turn = float(token[32]) * 2 - 1
    previous_thrust = float(token[33])
    previous_reward = float(token[34])

    left = max(0.0, -food_angle / math.pi) * food_strength
    right = max(0.0, food_angle / math.pi) * food_strength
    left *= 1.35
    right *= 1.35
    forward = (0.42 + max(0.0, math.cos(food_angle)) * 0.88) * (0.40 + food_strength * 0.72)
    brake = max(0.0, 0.18 - food_strength * 0.18)

    # Fast tactile reflex: if a wall is near, turn toward the open side and slow down.
    if max(forward_wall, left_wall, right_wall, edge) > 0.05:
        left += right_wall * 1.1 + forward_wall * (0.65 if left_wall < right_wall else 0.2) + edge * 0.2
        right += left_wall * 1.1 + forward_wall * (0.65 if right_wall <= left_wall else 0.2) + edge * 0.2
        forward -= forward_wall * 0.42 + edge * 0.18
        brake += forward_wall * 0.62 + edge * 0.25

    # Mushroom-body-like reward trace: repeat a recently rewarded movement, but only mildly.
    if previous_reward > 0.15:
        left += max(0, -previous_turn) * previous_reward * 0.24
        right += max(0, previous_turn) * previous_reward * 0.24
        forward += previous_thrust * previous_reward * 0.38

    return np.tanh(np.array([left, right, forward, brake], dtype=np.float32))


def make_contexts(seed=1204):
    rng = np.random.default_rng(seed)
    contexts = np.zeros((SAMPLE_COUNT, CONTEXT_LENGTH, FEATURE_COUNT), dtype=np.float32)
    labels = np.zeros((SAMPLE_COUNT, OUTPUT_COUNT), dtype=np.float32)

    for sample in range(SAMPLE_COUNT):
        target_angle = rng.uniform(-math.pi, math.pi)
        target_strength = rng.beta(1.2, 2.0)
        wall = rng.random(4) ** 2.8
        if rng.random() < 0.22:
            wall[rng.integers(0, 4)] = rng.uniform(0.55, 1.0)

        previous_turn = rng.uniform(-1, 1)
        previous_thrust = rng.uniform(0, 1)
        previous_reward = rng.choice([0.0, rng.uniform(0.25, 1.0)], p=[0.82, 0.18])
        energy = rng.uniform(0.25, 1.0)
        heading = rng.uniform(-math.pi, math.pi)
        position = rng.uniform(0, 1, 2)

        for time in range(CONTEXT_LENGTH):
            age = (time + 1) / CONTEXT_LENGTH
            token = np.zeros(FEATURE_COUNT, dtype=np.float32)
            drift = rng.normal(0, 0.18)
            for food_bin in range(16):
                bin_angle = (food_bin + 0.5) / 16 * 2 * math.pi - math.pi
                distance = abs(math.atan2(math.sin(bin_angle - target_angle - drift), math.cos(bin_angle - target_angle - drift)))
                token[food_bin] = target_strength * math.exp(-(distance ** 2) / 0.32) * (0.55 + age * 0.45)
            token[:16] += rng.normal(0, 0.018, 16)
            token[:16] = np.clip(token[:16], 0, 1)
            token[16] = rng.uniform(0.1, 1.0)
            token[17] = energy
            token[18] = math.sin(heading) * 0.5 + 0.5
            token[19] = math.cos(heading) * 0.5 + 0.5
            token[20:22] = position
            token[22:26] = np.clip(wall * (0.45 + age * 0.55) + rng.normal(0, 0.025, 4), 0, 1)
            token[26:32] = rng.uniform(0, 0.08, 6)
            token[32] = previous_turn * 0.5 + 0.5
            token[33] = previous_thrust
            token[34] = previous_reward
            token[35] = energy
            token[36] = 1.0
            token[37] = float(token[:16].max())
            token[38] = float(token[:16].mean())
            token[39] = 1.0
            contexts[sample, time] = token

        labels[sample] = expert_action(contexts[sample, -1])
    return contexts, labels


def transformer_embed(contexts, wq, wk, wv):
    q = contexts @ wq
    k = contexts @ wk
    v = contexts @ wv
    scores = q @ np.transpose(k, (0, 2, 1)) / math.sqrt(EMBED_DIM)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    attended = weights @ v
    return attended[:, -1, :]


def train_readout(embedding, labels):
    x = np.concatenate([embedding, np.ones((embedding.shape[0], 1), dtype=np.float32)], axis=1)
    y = np.arctanh(np.clip(labels, -0.94, 0.94))
    ridge = 0.08 * np.eye(x.shape[1], dtype=np.float32)
    solution = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    wo = solution[:-1].astype(np.float32)
    bo = solution[-1].astype(np.float32)
    pred = np.tanh(x @ solution)
    mae = float(np.mean(np.abs(pred - labels)))
    return wo, bo, pred, mae


def encode_sensors(agent, foods, scenario, rng, size=(900.0, 620.0)):
    x, y, heading, vx, vy = agent
    inputs = np.zeros(32, dtype=np.float32)
    sensor_range = scenario["sensor_range"]
    for fx, fy in foods:
        dx = fx - x
        dy = fy - y
        distance = math.hypot(dx, dy)
        if distance > sensor_range:
            continue
        angle = math.atan2(dy, dx) - heading
        angle = math.atan2(math.sin(angle), math.cos(angle))
        bin_index = int(np.clip(((angle + math.pi) / (2 * math.pi)) * 16, 0, 15))
        inputs[bin_index] = max(inputs[bin_index], 1.0 - distance / sensor_range)

    speed = math.hypot(vx, vy)
    inputs[16] = min(1.0, speed / scenario["max_speed"])
    inputs[17] = 0.8
    inputs[18] = math.sin(heading) * 0.5 + 0.5
    inputs[19] = math.cos(heading) * 0.5 + 0.5
    inputs[20] = x / size[0]
    inputs[21] = y / size[1]
    inputs[22] = wall_risk(x, y, heading, size)
    inputs[23] = wall_risk(x, y, heading - math.pi * 0.5, size)
    inputs[24] = wall_risk(x, y, heading + math.pi * 0.5, size)
    inputs[25] = edge_risk(x, y, size)
    inputs[26:32] = rng.uniform(0, 0.05, 6)
    return inputs


def wall_risk(x, y, angle, size):
    dx = math.cos(angle)
    dy = math.sin(angle)
    candidates = []
    if dx > 0:
        candidates.append((size[0] - x) / dx)
    elif dx < 0:
        candidates.append((0 - x) / dx)
    if dy > 0:
        candidates.append((size[1] - y) / dy)
    elif dy < 0:
        candidates.append((0 - y) / dy)
    distance = min([candidate for candidate in candidates if candidate > 0], default=130)
    return min(1.0, max(0.0, 1.0 - distance / 130))


def edge_risk(x, y, size):
    distance = min(x, y, size[0] - x, size[1] - y)
    return min(1.0, max(0.0, 1.0 - distance / 100))


def action_from_bias(bias):
    left, right, forward, brake = bias
    turn = float(np.clip(right - left, -1, 1))
    thrust = float(np.clip(0.18 + forward * 0.55 - brake * 0.32, 0, 1))
    return turn, thrust


def policy_bias(context, wq, wk, wv, wo, bo):
    embedding = transformer_embed(context[None, :, :], wq, wk, wv)[0]
    return np.tanh(embedding @ wo + bo)


def simulate_policy(wo, bo, wq, wk, wv, seed, episodes=7, steps=180):
    rng = np.random.default_rng(seed)
    total = 0.0
    size = (900.0, 620.0)
    for episode in range(episodes):
        scenario = {
            "food_count": int(rng.integers(12, 25)),
            "sensor_range": float(rng.uniform(210, 320)),
            "turn_rate": float(rng.uniform(5.0, 7.0)),
            "acceleration": float(rng.uniform(360, 500)),
            "damping": float(rng.uniform(0.925, 0.97)),
            "max_speed": float(rng.uniform(280, 380)),
        }
        foods = rng.uniform([34, 34], [size[0] - 34, size[1] - 34], (scenario["food_count"], 2))
        foods[0] = np.array([
            np.clip(size[0] * 0.5 + rng.uniform(-190, 190), 34, size[0] - 34),
            np.clip(size[1] * 0.5 + rng.uniform(-150, 150), 34, size[1] - 34),
        ])
        agent = np.array([size[0] * 0.5, size[1] * 0.5, rng.uniform(-math.pi, math.pi), 0.0, 0.0], dtype=np.float32)
        context = np.zeros((CONTEXT_LENGTH, FEATURE_COUNT), dtype=np.float32)
        previous_turn = 0.0
        previous_thrust = 0.0
        previous_reward = 0.0
        for _ in range(steps):
            sensors = encode_sensors(agent, foods, scenario, rng, size)
            wall_risk_now = float(max(sensors[22], sensors[23], sensors[24], sensors[25]))
            token = np.zeros(FEATURE_COUNT, dtype=np.float32)
            token[:32] = sensors
            token[32] = previous_turn * 0.5 + 0.5
            token[33] = previous_thrust
            token[34] = max(0.0, previous_reward / 26)
            token[35] = 0.8
            token[36] = 1.0
            token[37] = float(sensors[:16].max())
            token[38] = float(sensors[:16].mean())
            token[39] = 1.0
            context[:-1] = context[1:]
            context[-1] = token

            bias = policy_bias(context, wq, wk, wv, wo, bo)
            turn, thrust = action_from_bias(bias)
            dt = 1 / 30
            old_nearest = float(np.min(np.hypot(foods[:, 0] - agent[0], foods[:, 1] - agent[1])))
            agent[2] += turn * dt * scenario["turn_rate"]
            agent[3] += math.cos(agent[2]) * thrust * dt * scenario["acceleration"]
            agent[4] += math.sin(agent[2]) * thrust * dt * scenario["acceleration"]
            agent[3] *= scenario["damping"]
            agent[4] *= scenario["damping"]
            speed = math.hypot(agent[3], agent[4])
            if speed > scenario["max_speed"]:
                agent[3] *= scenario["max_speed"] / speed
                agent[4] *= scenario["max_speed"] / speed

            agent[0] += agent[3] * dt
            agent[1] += agent[4] * dt
            reward = -0.008
            hit_wall = False
            margin = 12
            if agent[0] < margin:
                agent[0] = margin
                agent[3] = abs(agent[3]) * 0.35
                hit_wall = True
            elif agent[0] > size[0] - margin:
                agent[0] = size[0] - margin
                agent[3] = -abs(agent[3]) * 0.35
                hit_wall = True
            if agent[1] < margin:
                agent[1] = margin
                agent[4] = abs(agent[4]) * 0.35
                hit_wall = True
            elif agent[1] > size[1] - margin:
                agent[1] = size[1] - margin
                agent[4] = -abs(agent[4]) * 0.35
                hit_wall = True
            if hit_wall:
                reward -= 18.0

            distances = np.hypot(foods[:, 0] - agent[0], foods[:, 1] - agent[1])
            new_nearest = float(np.min(distances))
            reward += float(np.clip((old_nearest - new_nearest) * 0.04, -0.28, 0.42))
            speed_after = math.hypot(agent[3], agent[4])
            if wall_risk_now > 0.35 and speed_after < scenario["max_speed"] * 0.12:
                reward -= 5.5 * wall_risk_now
            if sensors[22] > 0.35 and thrust > 0.45:
                reward -= 2.8 * sensors[22] * thrust
            if wall_risk_now > 0.25 and abs(turn) < 0.18:
                reward -= 1.4 * wall_risk_now
            if wall_risk_now > 0.25 and speed_after > scenario["max_speed"] * 0.18:
                reward += 0.38 * wall_risk_now
            eaten = np.where(distances < 19)[0]
            if eaten.size:
                reward += 36.0
                foods[eaten[0]] = rng.uniform([34, 34], [size[0] - 34, size[1] - 34])

            total += reward
            previous_turn = turn
            previous_thrust = thrust
            previous_reward = reward
    return total / episodes


def reinforce_readout(wo, bo, wq, wk, wv):
    rng = np.random.default_rng(8080)
    mean = np.concatenate([wo.reshape(-1), bo])
    sigma = np.full_like(mean, 0.035)
    best = mean.copy()
    best_score = simulate_policy(wo, bo, wq, wk, wv, seed=11, episodes=8, steps=220)
    for iteration in range(28):
        candidates = []
        for _ in range(42):
            vector = mean + rng.normal(0, sigma)
            cand_wo = vector[:-OUTPUT_COUNT].reshape(EMBED_DIM, OUTPUT_COUNT).astype(np.float32)
            cand_bo = vector[-OUTPUT_COUNT:].astype(np.float32)
            score = simulate_policy(cand_wo, cand_bo, wq, wk, wv, seed=1_000 + iteration, episodes=6, steps=190)
            candidates.append((score, vector))
        candidates.sort(key=lambda item: item[0], reverse=True)
        elite = np.array([vector for _, vector in candidates[:8]])
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.006) * 0.82
        check_wo = mean[:-OUTPUT_COUNT].reshape(EMBED_DIM, OUTPUT_COUNT).astype(np.float32)
        check_bo = mean[-OUTPUT_COUNT:].astype(np.float32)
        check_score = simulate_policy(check_wo, check_bo, wq, wk, wv, seed=77, episodes=8, steps=220)
        if check_score > best_score:
            best_score = check_score
            best = mean.copy()
        print(f"rl_iter={iteration + 1} score={check_score:.3f} best={best_score:.3f}")
    return (
        best[:-OUTPUT_COUNT].reshape(EMBED_DIM, OUTPUT_COUNT).astype(np.float32),
        best[-OUTPUT_COUNT:].astype(np.float32),
        float(best_score),
    )


def save_coreml(wq, wk, wv, wo, bo, output):
    scale = np.float32(1.0 / math.sqrt(EMBED_DIM))

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, CONTEXT_LENGTH, FEATURE_COUNT), dtype=types.fp32)],
        opset_version=ct.target.iOS17,
    )
    def transformer_context(context):
        q = mb.matmul(x=context, y=mb.const(val=wq, name="wq"), name="query")
        k = mb.matmul(x=context, y=mb.const(val=wk, name="wk"), name="key")
        v = mb.matmul(x=context, y=mb.const(val=wv, name="wv"), name="value")
        kt = mb.transpose(x=k, perm=[0, 2, 1], name="key_transposed")
        scores = mb.matmul(x=q, y=kt, name="attention_scores")
        scaled = mb.mul(x=scores, y=scale, name="scaled_scores")
        weights = mb.softmax(x=scaled, axis=-1, name="attention_weights")
        attended = mb.matmul(x=weights, y=v, name="attended_tokens")
        last = mb.slice_by_index(
            x=attended,
            begin=[0, CONTEXT_LENGTH - 1, 0],
            end=[1, CONTEXT_LENGTH, EMBED_DIM],
            name="last_token",
        )
        flat = mb.reshape(x=last, shape=[1, EMBED_DIM], name="context_embedding")
        logits = mb.linear(x=flat, weight=wo.T, bias=bo, name="action_logits")
        action_bias = mb.tanh(x=logits, name="actionBias")
        return action_bias

    model = ct.convert(
        transformer_context,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
    )
    model.short_description = "Insect-inspired Transformer policy trained on randomized foraging environments."
    model.input_description["context"] = "Recent insect-inspired sensor/action/reward context, shape 1 x 24 x 40."
    model.output_description["actionBias"] = "Four action bias values for left, right, forward, brake."
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))


def main():
    root = Path(__file__).resolve().parents[1]
    model_path = root / "NeuroGameMobile/Models/NeuroContextTransformer.mlpackage"
    report_path = root / "NeuroGameMobile/Models/InsectTrainingReport.json"

    wq, wk, wv = make_projection()
    contexts, labels = make_contexts()
    split = int(SAMPLE_COUNT * 0.82)
    train_embedding = transformer_embed(contexts[:split], wq, wk, wv)
    valid_embedding = transformer_embed(contexts[split:], wq, wk, wv)
    wo, bo, train_pred, train_mae = train_readout(train_embedding, labels[:split])
    wo, bo, rl_score = reinforce_readout(wo, bo, wq, wk, wv)
    valid_x = np.concatenate([valid_embedding, np.ones((valid_embedding.shape[0], 1), dtype=np.float32)], axis=1)
    valid_solution = np.concatenate([wo, bo.reshape(1, -1)], axis=0)
    valid_pred = np.tanh(valid_x @ valid_solution)
    valid_mae = float(np.mean(np.abs(valid_pred - labels[split:])))

    save_coreml(wq, wk, wv, wo, bo, model_path)
    report = {
        "training_samples": int(split),
        "validation_samples": int(SAMPLE_COUNT - split),
        "train_mae": train_mae,
        "validation_mae": valid_mae,
        "rl_score": rl_score,
        "architecture": "olfactory projection + central-complex heading + wall-risk sensors + reward/punishment reinforcement-tuned transformer readout",
        "outputs": ["left", "right", "forward", "brake"],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(model_path)


if __name__ == "__main__":
    main()
