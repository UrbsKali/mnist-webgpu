struct MatMulInfo {
  m : u32,
  n : u32,
  k : u32
}

@group(0) @binding(0) var<storage, read> aMatrix : array<f32>;
@group(0) @binding(1) var<storage, read> bMatrix : array<f32>;
@group(0) @binding(2) var<storage, read> biasVector : array<f32>;
@group(0) @binding(3) var<storage, read_write> result : array<f32>;
@group(0) @binding(4) var<uniform> info : MatMulInfo;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let index = global_id.x;
  let total = info.m * info.n;
  if (index >= total) {
    return;
  }

  let row = index / info.n;
  let col = index % info.n;

  var sum : f32 = 0.0;
  for (var kk : u32 = 0u; kk < info.k; kk = kk + 1u) {
    let aIdx = row * info.k + kk;
    let bIdx = kk * info.n + col;
    sum = sum + aMatrix[aIdx] * bMatrix[bIdx];
  }
  result[index] = sum + biasVector[col];
}
