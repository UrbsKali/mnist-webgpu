struct UnflattenInfo {
  width : u32,
  height : u32,
  channels : u32,
  features : u32,
  batch : u32
}

@group(0) @binding(0) var<storage, read> inputTensor : array<f32>; // [batch, features]
@group(0) @binding(1) var<storage, read_write> outputTensor : array<f32>; // [batch, height, width, channels]
@group(0) @binding(2) var<uniform> info : UnflattenInfo;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  let total = info.batch * info.features;
  if (index >= total) {
    return;
  }
  let batchIndex = index / info.features;
  let featureIndex = index % info.features;
  let channel = featureIndex % info.channels;
  let spatial = featureIndex / info.channels;
  let y = spatial / info.width;
  let x = spatial % info.width;

  let outputIndex = (((batchIndex * info.height + y) * info.width + x) * info.channels) + channel;
  outputTensor[outputIndex] = inputTensor[index];
}
