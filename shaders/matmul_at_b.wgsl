struct MatMulAtInfo {
  m : u32, // batch
  k : u32, // features
  n : u32 // classes
}

@group(0) @binding(0) var<storage, read> aMatrix : array<f32>; // [m, k]
@group(0) @binding(1) var<storage, read> bMatrix : array<f32>; // [m, n]
@group(0) @binding(2) var<storage, read_write> result : array<f32>; // [k, n]
@group(0) @binding(3) var<uniform> info : MatMulAtInfo;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let index = global_id.x;
  let total = info.k * info.n;
  if (index >= total) {
    return;
  }

  let row = index / info.n; // feature index
  let col = index % info.n; // class index

  var sum : f32 = 0.0;
  for (var batch : u32 = 0u; batch < info.m; batch = batch + 1u) {
    let aIdx = batch * info.k + row;
    let bIdx = batch * info.n + col;
    sum = sum + aMatrix[aIdx] * bMatrix[bIdx];
  }
  result[index] = sum;
}
