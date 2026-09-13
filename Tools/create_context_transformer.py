from pathlib import Path
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

rng = np.random.default_rng(20260913)
scale = np.float32(1.0 / math.sqrt(EMBED_DIM))

wq = rng.normal(0, 0.16, (FEATURE_COUNT, EMBED_DIM)).astype(np.float32)
wk = rng.normal(0, 0.16, (FEATURE_COUNT, EMBED_DIM)).astype(np.float32)
wv = rng.normal(0, 0.16, (FEATURE_COUNT, EMBED_DIM)).astype(np.float32)
wo = rng.normal(0, 0.18, (EMBED_DIM, OUTPUT_COUNT)).astype(np.float32)
bo = np.array([0.0, 0.0, 0.12, -0.08], dtype=np.float32)


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
    output = mb.tanh(x=logits, name="actionBias")
    return output


def main():
    model = ct.convert(
        transformer_context,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
    )
    model.short_description = "Tiny on-device transformer context model for NeuroGameMobile."
    model.input_description["context"] = "Recent sensor/action/reward tokens with shape 1 x 24 x 40."
    model.output_description["actionBias"] = "Four action bias values for left, right, forward, brake."

    output = Path(__file__).resolve().parents[1] / "NeuroGameMobile/Models/NeuroContextTransformer.mlpackage"
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    print(output)


if __name__ == "__main__":
    main()
