const MAX_CLASSES : u32 = 16u;

struct SoftmaxInfo {
  batchSize : u32,
  numClasses : u32,
  epsilon : f32,
  _pad : f32
}

@group(0) @binding(0) var<storage, read> logits : array<f32>;
@group(0) @binding(1) var<storage, read> labels : array<f32>;
@group(0) @binding(2) var<storage, read_write> probabilities : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradLogits : array<f32>;
@group(0) @binding(4) var<storage, read_write> losses : array<f32>;
@group(0) @binding(5) var<uniform> info : SoftmaxInfo;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let sample = global_id.x;
  if (sample >= info.batchSize) {
    return;
  }

  var maxLogit : f32 = -0x1.fffffep+127;
  let rowOffset = sample * info.numClasses;
  for (var c : u32 = 0u; c < info.numClasses; c = c + 1u) {
    let l = logits[rowOffset + c];
    if (l > maxLogit) {
      maxLogit = l;
    }
  }

  var sumExp : f32 = 0.0;
  var temp : array<f32, MAX_CLASSES>;
  for (var c : u32 = 0u; c < info.numClasses; c = c + 1u) {
    let shifted = logits[rowOffset + c] - maxLogit;
    let e = exp(shifted);
    temp[c] = e;
    sumExp = sumExp + e;
  }

  var loss : f32 = 0.0;
  for (var c : u32 = 0u; c < info.numClasses; c = c + 1u) {
    let prob = temp[c] / sumExp;
    probabilities[rowOffset + c] = prob;
    let label = labels[rowOffset + c];
    gradLogits[rowOffset + c] = prob - label;
    if (label > 0.5) {
      loss = -log(max(prob, info.epsilon));
    }
  }
  losses[sample] = loss;
}
