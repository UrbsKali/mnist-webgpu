struct ReduceInfo {
  batch : u32,
  components : u32
}

@group(0) @binding(0) var<storage, read> source : array<f32>;
@group(0) @binding(1) var<storage, read_write> dest : array<f32>;
@group(0) @binding(2) var<uniform> info : ReduceInfo;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let component = global_id.x;
  if (component >= info.components) {
    return;
  }
  var sum : f32 = 0.0;
  for (var batch : u32 = 0u; batch < info.batch; batch = batch + 1u) {
    let idx = batch * info.components + component;
    sum = sum + source[idx];
  }
  dest[component] = sum;
}
