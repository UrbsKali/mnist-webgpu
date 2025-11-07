struct ZeroInfo {
  size : u32
}

@group(0) @binding(0) var<storage, read_write> bufferData : array<f32>;
@group(0) @binding(1) var<uniform> info : ZeroInfo;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let idx = global_id.x;
  if (idx >= info.size) {
    return;
  }
  bufferData[idx] = 0.0;
}
