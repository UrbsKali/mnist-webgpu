struct PoolBackInfo {
  total : u32
}

@group(0) @binding(0) var<storage, read> gradOutput : array<f32>;
@group(0) @binding(1) var<storage, read> maskTensor : array<u32>;
@group(0) @binding(2) var<storage, read_write> gradInput : array<f32>;
@group(0) @binding(3) var<uniform> info : PoolBackInfo;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= info.total) {
    return;
  }
  let inputIndex = maskTensor[idx];
  let grad = gradOutput[idx];
  gradInput[inputIndex] = gradInput[inputIndex] + grad;
}
