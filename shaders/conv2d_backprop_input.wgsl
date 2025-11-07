struct Conv2DBackInputInfo {
  width : u32,
  height : u32,
  inChannels : u32,
  outChannels : u32,
  kernelSize : u32,
  batch : u32
}

@group(0) @binding(0) var<storage, read> gradOutput : array<f32>; // [b, h, w, outC]
@group(0) @binding(1) var<storage, read> filters : array<f32>; // [kH, kW, inC, outC]
@group(0) @binding(2) var<storage, read_write> gradInput : array<f32>; // [b, h, w, inC]
@group(0) @binding(3) var<uniform> info : Conv2DBackInputInfo;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let total = info.batch * info.height * info.width * info.inChannels;
  let index = gid.x;
  if (index >= total) {
    return;
  }

  let hw = info.height * info.width;
  let batchIndex = index / (hw * info.inChannels);
  let rem0 = index % (hw * info.inChannels);
  let spatialIndex = rem0 / info.inChannels;
  let channelIn = rem0 % info.inChannels;
  let y = spatialIndex / info.width;
  let x = spatialIndex % info.width;

  var acc : f32 = 0.0;
  let halfKernel = info.kernelSize / 2u;

  for (var ky : u32 = 0u; ky < info.kernelSize; ky = ky + 1u) {
    let outY = i32(y) - i32(ky) + i32(halfKernel);
    if (outY < 0 || outY >= i32(info.height)) {
      continue;
    }
    for (var kx : u32 = 0u; kx < info.kernelSize; kx = kx + 1u) {
      let outX = i32(x) - i32(kx) + i32(halfKernel);
      if (outX < 0 || outX >= i32(info.width)) {
        continue;
      }
      for (var oc : u32 = 0u; oc < info.outChannels; oc = oc + 1u) {
        let gradIndex = (((batchIndex * info.height + u32(outY)) * info.width + u32(outX)) * info.outChannels) + oc;
        let filterIndex = (((ky * info.kernelSize + kx) * info.inChannels + channelIn) * info.outChannels) + oc;
        acc = acc + gradOutput[gradIndex] * filters[filterIndex];
      }
    }
  }

  let inputIndex = (((batchIndex * info.height + y) * info.width + x) * info.inChannels) + channelIn;
  gradInput[inputIndex] = acc;
}
