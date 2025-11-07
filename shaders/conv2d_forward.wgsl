struct Conv2DInfo {
  width : u32,
  height : u32,
  inChannels : u32,
  outChannels : u32,
  kernelSize : u32,
  stride : u32,
  padding : u32,
  batch : u32
}

@group(0) @binding(0) var<storage, read> inputTensor : array<f32>; // NHWC
@group(0) @binding(1) var<storage, read> filterTensor : array<f32>; // [kH, kW, inC, outC]
@group(0) @binding(2) var<storage, read> biasTensor : array<f32>; // [outC]
@group(0) @binding(3) var<storage, read_write> outputTensor : array<f32>; // NHWC
@group(0) @binding(4) var<uniform> info : Conv2DInfo;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let total = info.batch * info.outChannels * info.width * info.height;
  let index = gid.x;
  if (index >= total) {
    return;
  }

  let spatial = info.width * info.height;
  let batchIndex = index / (info.outChannels * spatial);
  let rem0 = index % (info.outChannels * spatial);
  let channelOut = rem0 / spatial;
  let spatialIndex = rem0 % spatial;
  let y = spatialIndex / info.width;
  let x = spatialIndex % info.width;

  var acc : f32 = biasTensor[channelOut];
  let halfKernel = info.kernelSize / 2u;

  for (var ky : u32 = 0u; ky < info.kernelSize; ky = ky + 1u) {
    let inY = i32(y) + i32(ky) - i32(halfKernel);
    if (inY < 0 || inY >= i32(info.height)) {
      continue;
    }
    for (var kx : u32 = 0u; kx < info.kernelSize; kx = kx + 1u) {
      let inX = i32(x) + i32(kx) - i32(halfKernel);
      if (inX < 0 || inX >= i32(info.width)) {
        continue;
      }
      for (var ic : u32 = 0u; ic < info.inChannels; ic = ic + 1u) {
        let inputIndex = (((batchIndex * info.height + u32(inY)) * info.width + u32(inX)) * info.inChannels) + ic;
        let filterIndex = (((ky * info.kernelSize + kx) * info.inChannels + ic) * info.outChannels) + channelOut;
        acc = acc + inputTensor[inputIndex] * filterTensor[filterIndex];
      }
    }
  }

  let outputIndex = (((batchIndex * info.height + y) * info.width + x) * info.outChannels) + channelOut;
  outputTensor[outputIndex] = acc;
}
