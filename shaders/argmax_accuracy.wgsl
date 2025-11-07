struct AccuracyInfo {
  batchSize : u32,
  numClasses : u32
}

@group(0) @binding(0) var<storage, read> probabilities : array<f32>;
@group(0) @binding(1) var<storage, read> labels : array<f32>;
@group(0) @binding(2) var<storage, read_write> mask : array<u32>;
@group(0) @binding(3) var<uniform> info : AccuracyInfo;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let sample = global_id.x;
  if (sample >= info.batchSize) {
    return;
  }

  let rowOffset = sample * info.numClasses;
  var maxProb : f32 = -1.0;
  var argMax : u32 = 0u;
  for (var c : u32 = 0u; c < info.numClasses; c = c + 1u) {
    let value = probabilities[rowOffset + c];
    if (value > maxProb) {
      maxProb = value;
      argMax = c;
    }
  }

  var trueClass : u32 = 0u;
  for (var c : u32 = 0u; c < info.numClasses; c = c + 1u) {
    if (labels[rowOffset + c] > 0.5) {
      trueClass = c;
      break;
    }
  }

  mask[sample] = select(0u, 1u, argMax == trueClass);
}
