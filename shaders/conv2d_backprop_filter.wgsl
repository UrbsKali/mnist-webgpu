struct Conv2DFilterInfo {
  width : u32,
  height : u32,
  inChannels : u32,
  outChannels : u32,
  kernelSize : u32,
  stride : u32,
  padding : u32,
  batch : u32
}

@group(0) @binding(0) var<storage, read> inputTensor : array<f32>; // [b, h, w, inC]
@group(0) @binding(1) var<storage, read> gradOutput : array<f32>; // [b, h, w, outC]
@group(0) @binding(2) var<storage, read_write> gradFilter : array<f32>; // [kH, kW, inC, outC]
@group(0) @binding(3) var<uniform> info : Conv2DFilterInfo;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let total = info.kernelSize * info.kernelSize * info.inChannels * info.outChannels;
  let index = gid.x;
  if (index >= total) {
    return;
  }

  let perOut = info.kernelSize * info.kernelSize * info.inChannels;
  let outChannel = index / perOut;
  let rem0 = index % perOut;
  let perIn = info.kernelSize * info.kernelSize;
  let inChannel = rem0 / perIn;
  let kernelIndex = rem0 % perIn;
  let ky = kernelIndex / info.kernelSize;
  let kx = kernelIndex % info.kernelSize;
  let halfKernel = info.kernelSize / 2u;
  let stride = info.stride;
  let padding = info.padding;

  var acc : f32 = 0.0;
  for (var batch : u32 = 0u; batch < info.batch; batch = batch + 1u) {
    for (var y : u32 = 0u; y < info.height; y = y + 1u) {
      for (var x : u32 = 0u; x < info.width; x = x + 1u) {
        let inY = i32(y * stride) + i32(ky) - i32(padding);
        let inX = i32(x * stride) + i32(kx) - i32(padding);
        if (inY < 0 || inY >= i32(info.height) || inX < 0 || inX >= i32(info.width)) {
          continue;
        }
        let inputIndex = (((batch * info.height + u32(inY)) * info.width + u32(inX)) * info.inChannels) + inChannel;
        let gradIndex = (((batch * info.height + y) * info.width + x) * info.outChannels) + outChannel;
        acc = acc + inputTensor[inputIndex] * gradOutput[gradIndex];
      }
    }
  }

  let filterIndex = (((ky * info.kernelSize + kx) * info.inChannels + inChannel) * info.outChannels) + outChannel;
  gradFilter[filterIndex] = acc;
}
