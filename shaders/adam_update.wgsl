struct AdamInfo {
  learningRate : f32,
  beta1 : f32,
  beta2 : f32,
  epsilon : f32,
  oneMinusBeta1 : f32,
  oneMinusBeta2 : f32,
  beta1Power : f32,
  beta2Power : f32,
  size : u32
}

@group(0) @binding(0) var<storage, read_write> params : array<f32>;
@group(0) @binding(1) var<storage, read_write> grads : array<f32>;
@group(0) @binding(2) var<storage, read_write> mBuffer : array<f32>;
@group(0) @binding(3) var<storage, read_write> vBuffer : array<f32>;
@group(0) @binding(4) var<uniform> info : AdamInfo;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let idx = global_id.x;
  if (idx >= info.size) {
    return;
  }

  let grad = grads[idx];
  var m = mBuffer[idx];
  var v = vBuffer[idx];

  m = info.beta1 * m + info.oneMinusBeta1 * grad;
  v = info.beta2 * v + info.oneMinusBeta2 * (grad * grad);

  let mHat = m / (1.0 - info.beta1Power);
  let vHat = v / (1.0 - info.beta2Power);
  params[idx] = params[idx] - info.learningRate * mHat / (sqrt(vHat) + info.epsilon);

  mBuffer[idx] = m;
  vBuffer[idx] = v;
}
