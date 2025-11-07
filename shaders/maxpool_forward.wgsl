struct PoolInfo {
  inWidth : u32,
  inHeight : u32,
  channels : u32,
  window : u32,
  stride : u32,
  batch : u32,
  outWidth : u32,
  outHeight : u32
}

@group(0) @binding(0) var<storage, read> inputTensor : array<f32>;
@group(0) @binding(1) var<storage, read_write> outputTensor : array<f32>;
@group(0) @binding(2) var<storage, read_write> maskTensor : array<u32>;
@group(0) @binding(3) var<uniform> info : PoolInfo;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let total = info.batch * info.outWidth * info.outHeight * info.channels;
  let index = gid.x;
  if (index >= total) {
    return;
  }
  let spatialOut = info.outWidth * info.outHeight;
  let batchIndex = index / (spatialOut * info.channels);
  let rem0 = index % (spatialOut * info.channels);
  let spatialIndex = rem0 / info.channels;
  let channel = rem0 % info.channels;
  let outY = spatialIndex / info.outWidth;
  let outX = spatialIndex % info.outWidth;

  var bestValue : f32 = -0x1.fffffep+127;
  var bestIndex : u32 = 0u;

  for (var wy : u32 = 0u; wy < info.window; wy = wy + 1u) {
    let inY = outY * info.stride + wy;
    if (inY >= info.inHeight) {
      continue;
    }
    for (var wx : u32 = 0u; wx < info.window; wx = wx + 1u) {
      let inX = outX * info.stride + wx;
      if (inX >= info.inWidth) {
        continue;
      }
      let inputIndex = (((batchIndex * info.inHeight + inY) * info.inWidth + inX) * info.channels) + channel;
      let value = inputTensor[inputIndex];
      if (value > bestValue) {
        bestValue = value;
        bestIndex = inputIndex;
      }
    }
  }

  let outIndex = (((batchIndex * info.outHeight + outY) * info.outWidth + outX) * info.channels) + channel;
  outputTensor[outIndex] = bestValue;
  maskTensor[outIndex] = bestIndex;
}
