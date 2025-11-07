struct FlattenInfo {
  width : u32,
  height : u32,
  channels : u32,
  features : u32,
  batch : u32
}

@group(0) @binding(0) var<storage, read> inputTensor : array<f32>;
@group(0) @binding(1) var<storage, read_write> outputTensor : array<f32>;
@group(0) @binding(2) var<uniform> info : FlattenInfo;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  let total = info.batch * info.features;
  if (index >= total) {
    return;
  }
  let features = info.features;
  let batchIndex = index / features;
  let featureIndex = index % features;
  let channel = featureIndex % info.channels;
  let spatial = featureIndex / info.channels;
  let y = spatial / info.width;
  let x = spatial % info.width;

  let inputIndex = (((batchIndex * info.height + y) * info.width + x) * info.channels) + channel;
  outputTensor[index] = inputTensor[inputIndex];
}
