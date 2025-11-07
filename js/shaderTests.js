import { createBuffer, createEmptyBuffer, writeBuffer, ceilDiv } from './wgpu.js';

const FLOAT_TOLERANCE = 1e-4;
const STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const BYTES_F32 = 4;
const BYTES_U32 = 4;

function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function packUniform(entries) {
  const rawBytes = entries.length * 4;
  const byteLength = Math.max(16, alignTo(rawBytes, 16));
  const buffer = new ArrayBuffer(byteLength);
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);
  entries.forEach((entry, index) => {
    if (entry.type === 'f32') {
      f32[index] = entry.value;
    } else if (entry.type === 'u32') {
      u32[index] = entry.value >>> 0;
    } else {
      throw new Error(`Unsupported uniform type: ${entry.type}`);
    }
  });
  return buffer;
}

async function makeUniformBuffer(device, entries, label) {
  const data = packUniform(entries);
  const buffer = createEmptyBuffer(device, data.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label);
  await writeBuffer(device, buffer, data);
  return buffer;
}

async function makeStorageBuffer(device, values, label) {
  return createBuffer(device, values, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label);
}

function makeZeroFloatBuffer(device, length, label) {
  return createEmptyBuffer(device, length * BYTES_F32, STORAGE_USAGE, label);
}

function makeZeroUintBuffer(device, length, label) {
  return createEmptyBuffer(device, length * BYTES_U32, STORAGE_USAGE, label);
}

function computeAdamExpected(params, grads, mInit, vInit, info) {
  const size = params.length;
  const updatedParams = new Array(size);
  const updatedM = new Array(size);
  const updatedV = new Array(size);
  for (let i = 0; i < size; i += 1) {
    const grad = grads[i];
    const m = info.beta1 * mInit[i] + info.oneMinusBeta1 * grad;
    const v = info.beta2 * vInit[i] + info.oneMinusBeta2 * grad * grad;
    const mHat = m / (1 - info.beta1Power);
    const vHat = v / (1 - info.beta2Power);
    const param = params[i] - info.learningRate * mHat / (Math.sqrt(vHat) + info.epsilon);
    updatedParams[i] = param;
    updatedM[i] = m;
    updatedV[i] = v;
  }
  return { params: updatedParams, m: updatedM, v: updatedV };
}

function computeSoftmaxExpected(logits, labels, batchSize, numClasses, epsilon) {
  const probabilities = new Array(batchSize * numClasses);
  const gradients = new Array(batchSize * numClasses);
  const losses = new Array(batchSize);
  for (let sample = 0; sample < batchSize; sample += 1) {
    const rowOffset = sample * numClasses;
    let maxLogit = -Number.MAX_VALUE;
    for (let c = 0; c < numClasses; c += 1) {
      const value = logits[rowOffset + c];
      if (value > maxLogit) {
        maxLogit = value;
      }
    }
    let sumExp = 0;
    const temp = new Array(numClasses);
    for (let c = 0; c < numClasses; c += 1) {
      const shifted = logits[rowOffset + c] - maxLogit;
      const expValue = Math.exp(shifted);
      temp[c] = expValue;
      sumExp += expValue;
    }
    let loss = 0;
    for (let c = 0; c < numClasses; c += 1) {
      const prob = temp[c] / sumExp;
      probabilities[rowOffset + c] = prob;
      const label = labels[rowOffset + c];
      gradients[rowOffset + c] = prob - label;
      if (label > 0.5) {
        loss = -Math.log(Math.max(prob, epsilon));
      }
    }
    losses[sample] = loss;
  }
  return { probabilities, gradients, losses };
}

function multiplyMatrices(a, b, m, k, n, transposeSecond = false) {
  const result = new Array(m * n).fill(0);
  for (let row = 0; row < m; row += 1) {
    for (let col = 0; col < n; col += 1) {
      let sum = 0;
      for (let kk = 0; kk < k; kk += 1) {
        const aIdx = row * k + kk;
        const bIdx = transposeSecond ? col * k + kk : kk * n + col;
        sum += a[aIdx] * b[bIdx];
      }
      result[row * n + col] = sum;
    }
  }
  return result;
}

export const shaderTests = [
  {
    id: 'zero_buffer',
    label: 'zero_buffer.wgsl',
    description: 'Clears a storage buffer to zero using ZeroInfo.size elements.',
    workgroupSize: 128,
    shader: 'shaders/zero_buffer.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const size = 4;
      const data = new Float32Array([1, -2, 3, 4]);
      const dataBuffer = await makeStorageBuffer(device, data, 'debug-zero-data');
      buffers.push(dataBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: size },
      ], 'debug-zero-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: dataBuffer } },
          { binding: 1, resource: { buffer: uniformBuffer } },
        ],
      });

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(size, 128) },
        outputs: [
          {
            name: 'bufferData',
            buffer: dataBuffer,
            ctor: Float32Array,
            length: size,
            expected: new Array(size).fill(0),
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'scale_buffer',
    label: 'scale_buffer.wgsl',
    description: 'Scales each element in a buffer by a constant factor.',
    workgroupSize: 128,
    shader: 'shaders/scale_buffer.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const values = new Float32Array([2, -4, 6]);
      const factor = 0.5;
      const size = values.length;
      const dataBuffer = await makeStorageBuffer(device, values, 'debug-scale-data');
      buffers.push(dataBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'f32', value: factor },
        { type: 'u32', value: size },
        { type: 'f32', value: 0 },
        { type: 'f32', value: 0 },
      ], 'debug-scale-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: dataBuffer } },
          { binding: 1, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = Array.from(values, (v) => v * factor);

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(size, 128) },
        outputs: [
          {
            name: 'scaledBuffer',
            buffer: dataBuffer,
            ctor: Float32Array,
            length: size,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'relu_forward',
    label: 'relu_forward.wgsl',
    description: 'Applies ReLU activation element-wise.',
    workgroupSize: 128,
    shader: 'shaders/relu_forward.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const input = new Float32Array([-1, 0, 2, 3]);
      const inputBuffer = await makeStorageBuffer(device, input, 'debug-relu-in');
      buffers.push(inputBuffer);
      const outputBuffer = makeZeroFloatBuffer(device, input.length, 'debug-relu-out');
      buffers.push(outputBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
        ],
      });

      const expected = input.map((v) => (v > 0 ? v : 0));

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(input.length, 128) },
        outputs: [
          {
            name: 'reluOutput',
            buffer: outputBuffer,
            ctor: Float32Array,
            length: input.length,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'relu_backward',
    label: 'relu_backward.wgsl',
    description: 'Backpropagates gradients through the ReLU activation.',
    workgroupSize: 128,
    shader: 'shaders/relu_backward.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const gradOut = new Float32Array([1, 2, 3, 4]);
      const activations = new Float32Array([-1, 5, 0, 6]);
      const gradInputLength = gradOut.length;
      const gradOutBuffer = await makeStorageBuffer(device, gradOut, 'debug-relu-back-gradOut');
      buffers.push(gradOutBuffer);
      const activationsBuffer = await makeStorageBuffer(device, activations, 'debug-relu-back-acts');
      buffers.push(activationsBuffer);
      const gradInputBuffer = makeZeroFloatBuffer(device, gradInputLength, 'debug-relu-back-gradIn');
      buffers.push(gradInputBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: gradOutBuffer } },
          { binding: 1, resource: { buffer: activationsBuffer } },
          { binding: 2, resource: { buffer: gradInputBuffer } },
        ],
      });

      const expected = gradOut.map((value, idx) => (activations[idx] > 0 ? value : 0));

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(gradInputLength, 128) },
        outputs: [
          {
            name: 'reluGradInput',
            buffer: gradInputBuffer,
            ctor: Float32Array,
            length: gradInputLength,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'matmul_bias',
    label: 'matmul_bias.wgsl',
    description: 'Computes A·B + bias with row-major inputs.',
    workgroupSize: 128,
    shader: 'shaders/matmul_bias.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const m = 2;
      const k = 3;
      const n = 2;
      const a = new Float32Array([
        1, 2, 3,
        4, 5, 6,
      ]);
      const b = new Float32Array([
        1, 2,
        0, 1,
        1, 0,
      ]);
      const bias = new Float32Array([0.5, -1]);
      const resultLength = m * n;

      const aBuffer = await makeStorageBuffer(device, a, 'debug-matmul-bias-a');
      const bBuffer = await makeStorageBuffer(device, b, 'debug-matmul-bias-b');
      const biasBuffer = await makeStorageBuffer(device, bias, 'debug-matmul-bias-bias');
      const resultBuffer = makeZeroFloatBuffer(device, resultLength, 'debug-matmul-bias-out');
      buffers.push(aBuffer, bBuffer, biasBuffer, resultBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: m },
        { type: 'u32', value: n },
        { type: 'u32', value: k },
      ], 'debug-matmul-bias-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: aBuffer } },
          { binding: 1, resource: { buffer: bBuffer } },
          { binding: 2, resource: { buffer: biasBuffer } },
          { binding: 3, resource: { buffer: resultBuffer } },
          { binding: 4, resource: { buffer: uniformBuffer } },
        ],
      });

      const multiply = multiplyMatrices(a, b, m, k, n, false);
      const expected = multiply.map((value, idx) => value + bias[idx % n]);

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(resultLength, 128) },
        outputs: [
          {
            name: 'matmulBiasResult',
            buffer: resultBuffer,
            ctor: Float32Array,
            length: resultLength,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'softmax_cross_entropy_grad',
    label: 'softmax_cross_entropy_grad.wgsl',
    description: 'Computes softmax probabilities, gradients, and per-sample loss.',
    workgroupSize: 64,
    shader: 'shaders/softmax_cross_entropy_grad.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const batchSize = 2;
      const numClasses = 3;
      const logits = new Float32Array([
        1, 2, 0,
        0.5, 0.1, -0.3,
      ]);
      const labels = new Float32Array([
        0, 1, 0,
        1, 0, 0,
      ]);
      const probabilitiesBuffer = makeZeroFloatBuffer(device, batchSize * numClasses, 'debug-softmax-prob');
      const gradLogitsBuffer = makeZeroFloatBuffer(device, batchSize * numClasses, 'debug-softmax-grad');
      const lossBuffer = makeZeroFloatBuffer(device, batchSize, 'debug-softmax-loss');
      const logitsBuffer = await makeStorageBuffer(device, logits, 'debug-softmax-logits');
      const labelsBuffer = await makeStorageBuffer(device, labels, 'debug-softmax-labels');
      buffers.push(probabilitiesBuffer, gradLogitsBuffer, lossBuffer, logitsBuffer, labelsBuffer);

      const epsilon = 1e-7;
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: batchSize },
        { type: 'u32', value: numClasses },
        { type: 'f32', value: epsilon },
        { type: 'f32', value: 0 },
      ], 'debug-softmax-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: logitsBuffer } },
          { binding: 1, resource: { buffer: labelsBuffer } },
          { binding: 2, resource: { buffer: probabilitiesBuffer } },
          { binding: 3, resource: { buffer: gradLogitsBuffer } },
          { binding: 4, resource: { buffer: lossBuffer } },
          { binding: 5, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = computeSoftmaxExpected(Array.from(logits), Array.from(labels), batchSize, numClasses, epsilon);

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(batchSize, 64) },
        outputs: [
          {
            name: 'probabilities',
            buffer: probabilitiesBuffer,
            ctor: Float32Array,
            length: batchSize * numClasses,
            expected: expected.probabilities,
            tolerance: FLOAT_TOLERANCE,
          },
          {
            name: 'gradLogits',
            buffer: gradLogitsBuffer,
            ctor: Float32Array,
            length: batchSize * numClasses,
            expected: expected.gradients,
            tolerance: FLOAT_TOLERANCE,
          },
          {
            name: 'losses',
            buffer: lossBuffer,
            ctor: Float32Array,
            length: batchSize,
            expected: expected.losses,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'matmul_at_b',
    label: 'matmul_at_b.wgsl',
    description: 'Computes Aᵗ·B for batched matrices.',
    workgroupSize: 128,
    shader: 'shaders/matmul_at_b.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const m = 2;
      const k = 3;
      const n = 2;
      const a = new Float32Array([
        1, 2, 3,
        4, 5, 6,
      ]);
      const b = new Float32Array([
        1, 2,
        3, 0,
      ]);
      const resultLength = k * n;
      const aBuffer = await makeStorageBuffer(device, a, 'debug-matmul-atb-a');
      const bBuffer = await makeStorageBuffer(device, b, 'debug-matmul-atb-b');
      const resultBuffer = makeZeroFloatBuffer(device, resultLength, 'debug-matmul-atb-out');
      buffers.push(aBuffer, bBuffer, resultBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: m },
        { type: 'u32', value: k },
        { type: 'u32', value: n },
      ], 'debug-matmul-atb-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: aBuffer } },
          { binding: 1, resource: { buffer: bBuffer } },
          { binding: 2, resource: { buffer: resultBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = [];
      for (let row = 0; row < k; row += 1) {
        for (let col = 0; col < n; col += 1) {
          let sum = 0;
          for (let batch = 0; batch < m; batch += 1) {
            const aIdx = batch * k + row;
            const bIdx = batch * n + col;
            sum += a[aIdx] * b[bIdx];
          }
          expected.push(sum);
        }
      }

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(resultLength, 128) },
        outputs: [
          {
            name: 'matmulATB',
            buffer: resultBuffer,
            ctor: Float32Array,
            length: resultLength,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'matmul_abt',
    label: 'matmul_abt.wgsl',
    description: 'Computes A·Bᵗ.',
    workgroupSize: 128,
    shader: 'shaders/matmul_abt.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const m = 2;
      const k = 3;
      const n = 2;
      const a = new Float32Array([
        1, 2, 3,
        4, 5, 6,
      ]);
      const b = new Float32Array([
        1, 0, 1,
        0, 1, 1,
      ]);
      const resultLength = m * n;
      const aBuffer = await makeStorageBuffer(device, a, 'debug-matmul-abt-a');
      const bBuffer = await makeStorageBuffer(device, b, 'debug-matmul-abt-b');
      const resultBuffer = makeZeroFloatBuffer(device, resultLength, 'debug-matmul-abt-out');
      buffers.push(aBuffer, bBuffer, resultBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: m },
        { type: 'u32', value: n },
        { type: 'u32', value: k },
      ], 'debug-matmul-abt-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: aBuffer } },
          { binding: 1, resource: { buffer: bBuffer } },
          { binding: 2, resource: { buffer: resultBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = [];
      for (let row = 0; row < m; row += 1) {
        for (let col = 0; col < n; col += 1) {
          let sum = 0;
          for (let kk = 0; kk < k; kk += 1) {
            sum += a[row * k + kk] * b[col * k + kk];
          }
          expected.push(sum);
        }
      }

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(resultLength, 128) },
        outputs: [
          {
            name: 'matmulABT',
            buffer: resultBuffer,
            ctor: Float32Array,
            length: resultLength,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'reduce_sum_axis0',
    label: 'reduce_sum_axis0.wgsl',
    description: 'Reduces across batch dimension producing component sums.',
    workgroupSize: 64,
    shader: 'shaders/reduce_sum_axis0.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const batch = 3;
      const components = 2;
      const source = new Float32Array([
        1, 2,
        3, 4,
        5, 6,
      ]);
      const destLength = components;
      const sourceBuffer = await makeStorageBuffer(device, source, 'debug-reduce-source');
      const destBuffer = makeZeroFloatBuffer(device, destLength, 'debug-reduce-dest');
      buffers.push(sourceBuffer, destBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: batch },
        { type: 'u32', value: components },
      ], 'debug-reduce-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: sourceBuffer } },
          { binding: 1, resource: { buffer: destBuffer } },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = new Array(destLength).fill(0);
      for (let component = 0; component < components; component += 1) {
        let sum = 0;
        for (let b = 0; b < batch; b += 1) {
          sum += source[b * components + component];
        }
        expected[component] = sum;
      }

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(destLength, 64) },
        outputs: [
          {
            name: 'reduced',
            buffer: destBuffer,
            ctor: Float32Array,
            length: destLength,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'adam_update',
    label: 'adam_update.wgsl',
    description: 'Performs one Adam optimizer update step.',
    workgroupSize: 128,
    shader: 'shaders/adam_update.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const params = new Float32Array([0.1, -0.2, 0.3]);
      const grads = new Float32Array([0.01, -0.03, 0.05]);
      const mInit = new Float32Array([0.001, -0.002, 0.003]);
      const vInit = new Float32Array([0.0004, 0.0005, 0.0006]);
      const size = params.length;
      const info = {
        learningRate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        oneMinusBeta1: 0.1,
        oneMinusBeta2: 0.001,
        beta1Power: 0.9,
        beta2Power: 0.999,
        size,
      };

      const paramsBuffer = await makeStorageBuffer(device, params, 'debug-adam-params');
      const gradsBuffer = await makeStorageBuffer(device, grads, 'debug-adam-grads');
      const mBuffer = await makeStorageBuffer(device, mInit, 'debug-adam-m');
      const vBuffer = await makeStorageBuffer(device, vInit, 'debug-adam-v');
      buffers.push(paramsBuffer, gradsBuffer, mBuffer, vBuffer);

      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'f32', value: info.learningRate },
        { type: 'f32', value: info.beta1 },
        { type: 'f32', value: info.beta2 },
        { type: 'f32', value: info.epsilon },
        { type: 'f32', value: info.oneMinusBeta1 },
        { type: 'f32', value: info.oneMinusBeta2 },
        { type: 'f32', value: info.beta1Power },
        { type: 'f32', value: info.beta2Power },
        { type: 'u32', value: size },
      ], 'debug-adam-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: gradsBuffer } },
          { binding: 2, resource: { buffer: mBuffer } },
          { binding: 3, resource: { buffer: vBuffer } },
          { binding: 4, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = computeAdamExpected(Array.from(params), Array.from(grads), Array.from(mInit), Array.from(vInit), info);

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(size, 128) },
        outputs: [
          {
            name: 'adamParams',
            buffer: paramsBuffer,
            ctor: Float32Array,
            length: size,
            expected: expected.params,
            tolerance: FLOAT_TOLERANCE,
          },
          {
            name: 'adamM',
            buffer: mBuffer,
            ctor: Float32Array,
            length: size,
            expected: expected.m,
            tolerance: FLOAT_TOLERANCE,
          },
          {
            name: 'adamV',
            buffer: vBuffer,
            ctor: Float32Array,
            length: size,
            expected: expected.v,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'sgd_update',
    label: 'sgd_update.wgsl',
    description: 'Performs a single SGD parameter update.',
    workgroupSize: 128,
    shader: 'shaders/sgd_update.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const params = new Float32Array([0.2, -0.4, 0.6]);
      const grads = new Float32Array([0.5, -0.25, 0.125]);
      const learningRate = 0.05;
      const size = params.length;
      const paramsBuffer = await makeStorageBuffer(device, params, 'debug-sgd-params');
      const gradsBuffer = await makeStorageBuffer(device, grads, 'debug-sgd-grads');
      buffers.push(paramsBuffer, gradsBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'f32', value: learningRate },
        { type: 'u32', value: size },
        { type: 'f32', value: 0 },
        { type: 'f32', value: 0 },
      ], 'debug-sgd-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: gradsBuffer } },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = params.map((value, idx) => value - learningRate * grads[idx]);

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(size, 128) },
        outputs: [
          {
            name: 'sgdParams',
            buffer: paramsBuffer,
            ctor: Float32Array,
            length: size,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'argmax_accuracy',
    label: 'argmax_accuracy.wgsl',
    description: 'Computes prediction accuracy mask via argmax.',
    workgroupSize: 64,
    shader: 'shaders/argmax_accuracy.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const batchSize = 2;
      const numClasses = 3;
      const probabilities = new Float32Array([
        0.1, 0.7, 0.2,
        0.6, 0.1, 0.3,
      ]);
      const labels = new Float32Array([
        0, 1, 0,
        0, 0, 1,
      ]);
      const maskLength = batchSize;
      const probabilitiesBuffer = await makeStorageBuffer(device, probabilities, 'debug-acc-prob');
      const labelsBuffer = await makeStorageBuffer(device, labels, 'debug-acc-labels');
      const maskBuffer = makeZeroUintBuffer(device, maskLength, 'debug-acc-mask');
      buffers.push(probabilitiesBuffer, labelsBuffer, maskBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: batchSize },
        { type: 'u32', value: numClasses },
      ], 'debug-acc-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: probabilitiesBuffer } },
          { binding: 1, resource: { buffer: labelsBuffer } },
          { binding: 2, resource: { buffer: maskBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = [1, 0];

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(maskLength, 64) },
        outputs: [
          {
            name: 'accuracyMask',
            buffer: maskBuffer,
            ctor: Uint32Array,
            length: maskLength,
            expected,
            tolerance: 0,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'flatten',
    label: 'flatten.wgsl',
    description: 'Flattens NHWC tensor into [batch, features] layout.',
    workgroupSize: 128,
    shader: 'shaders/flatten.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const width = 2;
      const height = 2;
      const channels = 2;
      const batch = 1;
      const features = width * height * channels;
      const input = new Float32Array([
        1, 2,
        3, 4,
        5, 6,
        7, 8,
      ]);
      const inputBuffer = await makeStorageBuffer(device, input, 'debug-flatten-input');
      const outputBuffer = makeZeroFloatBuffer(device, batch * features, 'debug-flatten-output');
      buffers.push(inputBuffer, outputBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: width },
        { type: 'u32', value: height },
        { type: 'u32', value: channels },
        { type: 'u32', value: features },
        { type: 'u32', value: batch },
      ], 'debug-flatten-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = Array.from(input);

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(batch * features, 128) },
        outputs: [
          {
            name: 'flattenOutput',
            buffer: outputBuffer,
            ctor: Float32Array,
            length: batch * features,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'unflatten',
    label: 'unflatten.wgsl',
    description: 'Reshapes flat feature vectors back to NHWC layout.',
    workgroupSize: 128,
    shader: 'shaders/unflatten.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const width = 2;
      const height = 2;
      const channels = 2;
      const batch = 1;
      const features = width * height * channels;
      const input = new Float32Array([
        1, 2, 3, 4, 5, 6, 7, 8,
      ]);
      const inputBuffer = await makeStorageBuffer(device, input, 'debug-unflatten-input');
      const outputBuffer = makeZeroFloatBuffer(device, batch * width * height * channels, 'debug-unflatten-output');
      buffers.push(inputBuffer, outputBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: width },
        { type: 'u32', value: height },
        { type: 'u32', value: channels },
        { type: 'u32', value: features },
        { type: 'u32', value: batch },
      ], 'debug-unflatten-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = Array.from(input);

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(batch * features, 128) },
        outputs: [
          {
            name: 'unflattenOutput',
            buffer: outputBuffer,
            ctor: Float32Array,
            length: batch * width * height * channels,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'conv2d_forward',
    label: 'conv2d_forward.wgsl',
    description: 'Computes NHWC convolution with bias.',
    workgroupSize: 64,
    shader: 'shaders/conv2d_forward.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const info = {
        width: 2,
        height: 2,
        inChannels: 1,
        outChannels: 1,
        kernelSize: 1,
        stride: 1,
        padding: 0,
        batch: 1,
      };
      const input = new Float32Array([1, 2, 3, 4]);
      const filter = new Float32Array([0.5]);
      const bias = new Float32Array([0.1]);
      const total = info.batch * info.width * info.height * info.outChannels;
      const inputBuffer = await makeStorageBuffer(device, input, 'debug-conv-fwd-input');
      const filterBuffer = await makeStorageBuffer(device, filter, 'debug-conv-fwd-filter');
      const biasBuffer = await makeStorageBuffer(device, bias, 'debug-conv-fwd-bias');
      const outputBuffer = makeZeroFloatBuffer(device, total, 'debug-conv-fwd-output');
      buffers.push(inputBuffer, filterBuffer, biasBuffer, outputBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: info.width },
        { type: 'u32', value: info.height },
        { type: 'u32', value: info.inChannels },
        { type: 'u32', value: info.outChannels },
        { type: 'u32', value: info.kernelSize },
        { type: 'u32', value: info.stride },
        { type: 'u32', value: info.padding },
        { type: 'u32', value: info.batch },
      ], 'debug-conv-fwd-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: filterBuffer } },
          { binding: 2, resource: { buffer: biasBuffer } },
          { binding: 3, resource: { buffer: outputBuffer } },
          { binding: 4, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = input.map((value) => value * filter[0] + bias[0]);

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(total, 64) },
        outputs: [
          {
            name: 'convForward',
            buffer: outputBuffer,
            ctor: Float32Array,
            length: total,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'conv2d_backprop_filter',
    label: 'conv2d_backprop_filter.wgsl',
    description: 'Computes conv filter gradients using input activations and gradOutput.',
    workgroupSize: 64,
    shader: 'shaders/conv2d_backprop_filter.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const info = {
        width: 2,
        height: 2,
        inChannels: 1,
        outChannels: 1,
        kernelSize: 1,
        stride: 1,
        padding: 0,
        batch: 1,
      };
      const input = new Float32Array([1, 2, 3, 4]);
      const gradOutput = new Float32Array([0.1, 0.2, 0.3, 0.4]);
      const filterElements = info.kernelSize * info.kernelSize * info.inChannels * info.outChannels;
      const inputBuffer = await makeStorageBuffer(device, input, 'debug-conv-filter-input');
      const gradOutBuffer = await makeStorageBuffer(device, gradOutput, 'debug-conv-filter-gradOut');
      const gradFilterBuffer = makeZeroFloatBuffer(device, filterElements, 'debug-conv-filter-grad');
      buffers.push(inputBuffer, gradOutBuffer, gradFilterBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: info.width },
        { type: 'u32', value: info.height },
        { type: 'u32', value: info.inChannels },
        { type: 'u32', value: info.outChannels },
        { type: 'u32', value: info.kernelSize },
        { type: 'u32', value: info.stride },
        { type: 'u32', value: info.padding },
        { type: 'u32', value: info.batch },
      ], 'debug-conv-filter-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: gradOutBuffer } },
          { binding: 2, resource: { buffer: gradFilterBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer } },
        ],
      });

      let expected = 0;
      for (let i = 0; i < input.length; i += 1) {
        expected += input[i] * gradOutput[i];
      }

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(filterElements, 64) },
        outputs: [
          {
            name: 'convFilterGrad',
            buffer: gradFilterBuffer,
            ctor: Float32Array,
            length: filterElements,
            expected: [expected],
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'conv2d_backprop_input',
    label: 'conv2d_backprop_input.wgsl',
    description: 'Computes input gradients for convolution via filter weights.',
    workgroupSize: 64,
    shader: 'shaders/conv2d_backprop_input.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const info = {
        width: 2,
        height: 2,
        inChannels: 1,
        outChannels: 1,
        kernelSize: 1,
        batch: 1,
      };
      const gradOutput = new Float32Array([0.1, 0.2, 0.3, 0.4]);
      const filters = new Float32Array([0.5]);
      const gradInputLength = info.batch * info.width * info.height * info.inChannels;
      const gradOutBuffer = await makeStorageBuffer(device, gradOutput, 'debug-conv-input-gradOut');
      const filterBuffer = await makeStorageBuffer(device, filters, 'debug-conv-input-filter');
      const gradInputBuffer = makeZeroFloatBuffer(device, gradInputLength, 'debug-conv-input-grad');
      buffers.push(gradOutBuffer, filterBuffer, gradInputBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: info.width },
        { type: 'u32', value: info.height },
        { type: 'u32', value: info.inChannels },
        { type: 'u32', value: info.outChannels },
        { type: 'u32', value: info.kernelSize },
        { type: 'u32', value: info.batch },
      ], 'debug-conv-input-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: gradOutBuffer } },
          { binding: 1, resource: { buffer: filterBuffer } },
          { binding: 2, resource: { buffer: gradInputBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = Array.from(gradOutput, (value) => value * filters[0]);

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(gradInputLength, 64) },
        outputs: [
          {
            name: 'convInputGrad',
            buffer: gradInputBuffer,
            ctor: Float32Array,
            length: gradInputLength,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'maxpool_forward',
    label: 'maxpool_forward.wgsl',
    description: 'Performs 2×2 max pooling with index mask output.',
    workgroupSize: 64,
    shader: 'shaders/maxpool_forward.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const info = {
        inWidth: 2,
        inHeight: 2,
        channels: 1,
        window: 2,
        stride: 2,
        batch: 1,
        outWidth: 1,
        outHeight: 1,
      };
      const input = new Float32Array([1, 2, 4, 3]);
      const outputLength = info.batch * info.outWidth * info.outHeight * info.channels;
      const maskLength = outputLength;
      const inputBuffer = await makeStorageBuffer(device, input, 'debug-pool-forward-input');
      const outputBuffer = makeZeroFloatBuffer(device, outputLength, 'debug-pool-forward-output');
      const maskBuffer = makeZeroUintBuffer(device, maskLength, 'debug-pool-forward-mask');
      buffers.push(inputBuffer, outputBuffer, maskBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: info.inWidth },
        { type: 'u32', value: info.inHeight },
        { type: 'u32', value: info.channels },
        { type: 'u32', value: info.window },
        { type: 'u32', value: info.stride },
        { type: 'u32', value: info.batch },
        { type: 'u32', value: info.outWidth },
        { type: 'u32', value: info.outHeight },
      ], 'debug-pool-forward-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: maskBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer } },
        ],
      });

      const expectedValue = Math.max(...input);
      const expectedIndex = input.findIndex((value) => value === expectedValue);

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(outputLength, 64) },
        outputs: [
          {
            name: 'poolOutput',
            buffer: outputBuffer,
            ctor: Float32Array,
            length: outputLength,
            expected: [expectedValue],
            tolerance: FLOAT_TOLERANCE,
          },
          {
            name: 'poolMask',
            buffer: maskBuffer,
            ctor: Uint32Array,
            length: maskLength,
            expected: [expectedIndex >>> 0],
            tolerance: 0,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
  {
    id: 'maxpool_backward',
    label: 'maxpool_backward.wgsl',
    description: 'Scatters pooled gradients back to input tensor using mask indices.',
    workgroupSize: 64,
    shader: 'shaders/maxpool_backward.wgsl',
    async setup({ device, pipeline }) {
      const buffers = [];
      const total = 1;
      const gradOutput = new Float32Array([0.5]);
      const mask = new Uint32Array([2]);
      const gradInputLength = 4;
      const gradOutputBuffer = await makeStorageBuffer(device, gradOutput, 'debug-pool-back-gradOut');
      const maskBuffer = await makeStorageBuffer(device, mask, 'debug-pool-back-mask');
      const gradInputBuffer = makeZeroFloatBuffer(device, gradInputLength, 'debug-pool-back-gradIn');
      buffers.push(gradOutputBuffer, maskBuffer, gradInputBuffer);
      const uniformBuffer = await makeUniformBuffer(device, [
        { type: 'u32', value: total },
      ], 'debug-pool-back-uniform');
      buffers.push(uniformBuffer);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: gradOutputBuffer } },
          { binding: 1, resource: { buffer: maskBuffer } },
          { binding: 2, resource: { buffer: gradInputBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer } },
        ],
      });

      const expected = [0, 0, gradOutput[0], 0];

      return {
        bindGroups: [bindGroup],
        dispatch: { x: ceilDiv(total, 64) },
        outputs: [
          {
            name: 'poolGradInput',
            buffer: gradInputBuffer,
            ctor: Float32Array,
            length: gradInputLength,
            expected,
            tolerance: FLOAT_TOLERANCE,
          },
        ],
        destroy() {
          buffers.forEach((buffer) => buffer.destroy());
        },
      };
    },
  },
];

export const DEFAULT_FLOAT_TOLERANCE = FLOAT_TOLERANCE;
