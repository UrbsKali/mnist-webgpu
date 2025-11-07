import { loadShaderModule, createBuffer, createEmptyBuffer, writeBuffer, readBufferToArray, ceilDiv, createComputePipeline } from '../wgpu.js';
import { DataConstants } from '../data.js';

const WORKGROUP_128 = 128;
const WORKGROUP_64 = 64;

function encodeUniform(entries) {
  const buffer = new ArrayBuffer(entries.length * 4);
  const view = new DataView(buffer);
  for (let i = 0; i < entries.length; i += 1) {
    const entry = entries[i];
    const [type, value] = Array.isArray(entry) ? entry : [entry.type, entry.value];
    const offset = i * 4;
    if (type === 'u32') {
      view.setUint32(offset, value >>> 0, true);
    } else if (type === 'f32') {
      view.setFloat32(offset, Math.fround(value), true);
    } else {
      throw new Error(`Unsupported uniform field type "${type}"`);
    }
  }
  return new Uint8Array(buffer);
}

export class LogisticRegressionModel {
  constructor(device) {
    this.device = device;
    this.queue = device.queue;
    this.initialized = false;
    this.compiled = false;
    this.optimizer = 'adam';
    this.learningRate = 0.001;
    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.epsilon = 1e-8;
    this.beta1Power = 0.9;
    this.beta2Power = 0.999;
    this.step = 0;

    this.features = DataConstants.imageSize;
    this.classes = DataConstants.numClasses;

    this.uniformBuffers = {};
    this.bindGroups = {};
    this.pipelines = {};
  }

  async compile() {
    if (this.compiled) return;
    const forwardModule = await loadShaderModule(this.device, 'shaders/matmul_bias.wgsl');
    const softmaxModule = await loadShaderModule(this.device, 'shaders/softmax_cross_entropy_grad.wgsl');
    const gradModule = await loadShaderModule(this.device, 'shaders/matmul_at_b.wgsl');
    const reduceModule = await loadShaderModule(this.device, 'shaders/reduce_sum_axis0.wgsl');
    const scaleModule = await loadShaderModule(this.device, 'shaders/scale_buffer.wgsl');
    const sgdModule = await loadShaderModule(this.device, 'shaders/sgd_update.wgsl');
    const adamModule = await loadShaderModule(this.device, 'shaders/adam_update.wgsl');
    const accuracyModule = await loadShaderModule(this.device, 'shaders/argmax_accuracy.wgsl');

    this.pipelines.forward = await createComputePipeline(this.device, {
      label: 'lr-forward',
      layout: 'auto',
      compute: { module: forwardModule, entryPoint: 'main' },
    });
    this.pipelines.softmax = await createComputePipeline(this.device, {
      label: 'lr-softmax',
      layout: 'auto',
      compute: { module: softmaxModule, entryPoint: 'main' },
    });
    this.pipelines.gradWeights = await createComputePipeline(this.device, {
      label: 'lr-gradW',
      layout: 'auto',
      compute: { module: gradModule, entryPoint: 'main' },
    });
    this.pipelines.reduceBias = await createComputePipeline(this.device, {
      label: 'lr-reduce-bias',
      layout: 'auto',
      compute: { module: reduceModule, entryPoint: 'main' },
    });
    this.pipelines.scale = await createComputePipeline(this.device, {
      label: 'lr-scale',
      layout: 'auto',
      compute: { module: scaleModule, entryPoint: 'main' },
    });
    this.pipelines.sgd = await createComputePipeline(this.device, {
      label: 'lr-sgd',
      layout: 'auto',
      compute: { module: sgdModule, entryPoint: 'main' },
    });
    this.pipelines.adam = await createComputePipeline(this.device, {
      label: 'lr-adam',
      layout: 'auto',
      compute: { module: adamModule, entryPoint: 'main' },
    });
    this.pipelines.accuracy = await createComputePipeline(this.device, {
      label: 'lr-accuracy',
      layout: 'auto',
      compute: { module: accuracyModule, entryPoint: 'main' },
    });

    this.compiled = true;
  }

  async initialize({ batchSize, learningRate, optimizer, seed }) {
    await this.compile();
    this.batchSize = batchSize;
    this.learningRate = learningRate;
    this.optimizer = optimizer;
    this.step = 0;

    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.epsilon = 1e-8;
    this.beta1Power = this.beta1;
    this.beta2Power = this.beta2;

    this.uniformBuffers = {};
    this.bindGroups = {};

    const rng = mulberry32(seed ?? 42);
    const weightArray = new Float32Array(this.features * this.classes);
    const std = Math.sqrt(2 / (this.features + this.classes));
    for (let i = 0; i < weightArray.length; i += 1) {
      weightArray[i] = rng() * 2 * std - std;
    }
    const biasArray = new Float32Array(this.classes);

    this.weightBuffer = await createBuffer(
      this.device,
      weightArray,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      'lr-weights',
    );
    this.biasBuffer = await createBuffer(
      this.device,
      biasArray,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      'lr-bias',
    );
    this.gradWeightBuffer = createEmptyBuffer(this.device, weightArray.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'lr-gradW');
    this.gradBiasBuffer = createEmptyBuffer(this.device, biasArray.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'lr-gradB');

    this.inputBuffer = createEmptyBuffer(this.device, batchSize * this.features * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'lr-input');
    this.labelBuffer = createEmptyBuffer(this.device, batchSize * this.classes * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'lr-labels');
    this.logitsBuffer = createEmptyBuffer(this.device, batchSize * this.classes * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'lr-logits');
    this.probBuffer = createEmptyBuffer(this.device, batchSize * this.classes * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'lr-probs');
    this.gradLogitsBuffer = createEmptyBuffer(this.device, batchSize * this.classes * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'lr-gradLogits');
    this.lossBuffer = createEmptyBuffer(this.device, batchSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'lr-losses');
    this.accuracyMaskBuffer = createEmptyBuffer(this.device, batchSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'lr-acc-mask');

    this.mWeightBuffer = createEmptyBuffer(this.device, weightArray.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'lr-mW');
    this.vWeightBuffer = createEmptyBuffer(this.device, weightArray.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'lr-vW');
    this.mBiasBuffer = createEmptyBuffer(this.device, biasArray.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'lr-mB');
    this.vBiasBuffer = createEmptyBuffer(this.device, biasArray.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'lr-vB');

    this.uniformBuffers.forward = createEmptyBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'lr-forward-info');
    this.uniformBuffers.softmax = createEmptyBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'lr-softmax-info');
    this.uniformBuffers.grad = createEmptyBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'lr-grad-info');
    this.uniformBuffers.reduce = createEmptyBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'lr-reduce-info');
    this.uniformBuffers.scaleWeights = createEmptyBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'lr-scale-weights');
    this.uniformBuffers.scaleBias = createEmptyBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'lr-scale-bias');
    this.uniformBuffers.sgdWeights = createEmptyBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'lr-sgd-weights');
    this.uniformBuffers.sgdBias = createEmptyBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'lr-sgd-bias');
    this.uniformBuffers.adamWeights = createEmptyBuffer(this.device, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'lr-adam-weights');
    this.uniformBuffers.adamBias = createEmptyBuffer(this.device, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'lr-adam-bias');
    this.uniformBuffers.accuracy = createEmptyBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'lr-acc-info');

    this.initialized = true;
  }

  async updateUniformBuffer(buffer, data) {
    await writeBuffer(this.device, buffer, data);
  }

  async updateForwardUniforms(batch) {
    const info = new Uint32Array([batch, this.classes, this.features, 0]);
    await writeBuffer(this.device, this.uniformBuffers.forward, info);
    await writeBuffer(this.device, this.uniformBuffers.grad, new Uint32Array([batch, this.features, this.classes, 0]));
  }

  async updateSoftmaxUniforms(batch) {
    const data = encodeUniform([
      ['u32', batch],
      ['u32', this.classes],
      ['f32', 1e-7],
      ['f32', 0],
    ]);
    await writeBuffer(this.device, this.uniformBuffers.softmax, data);
    const acc = new Uint32Array([batch, this.classes, 0]);
    await writeBuffer(this.device, this.uniformBuffers.accuracy, acc);
  }

  async updateReduceUniforms(batch) {
    const info = new Uint32Array([batch, this.classes, 0, 0]);
    await writeBuffer(this.device, this.uniformBuffers.reduce, info);
  }

  async updateScaleUniform(buffer, size, factor) {
    const data = encodeUniform([
      ['f32', factor],
      ['u32', size],
      ['f32', 0],
      ['f32', 0],
    ]);
    await writeBuffer(this.device, buffer, data);
  }

  async updateSgdUniform(buffer, size) {
    const data = encodeUniform([
      ['f32', this.learningRate],
      ['u32', size],
      ['f32', 0],
      ['f32', 0],
    ]);
    await writeBuffer(this.device, buffer, data);
  }

  async updateAdamUniform(buffer, size) {
    const array = encodeUniform([
      ['f32', this.learningRate],
      ['f32', this.beta1],
      ['f32', this.beta2],
      ['f32', this.epsilon],
      ['f32', 1 - this.beta1],
      ['f32', 1 - this.beta2],
      ['f32', this.beta1Power],
      ['f32', this.beta2Power],
      ['u32', size],
      ['u32', 0],
      ['u32', 0],
      ['u32', 0],
    ]);
    await writeBuffer(this.device, buffer, array);
  }

  createBindGroups() {
    this.bindGroups.forward = this.device.createBindGroup({
      layout: this.pipelines.forward.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.inputBuffer } },
        { binding: 1, resource: { buffer: this.weightBuffer } },
        { binding: 2, resource: { buffer: this.biasBuffer } },
        { binding: 3, resource: { buffer: this.logitsBuffer } },
        { binding: 4, resource: { buffer: this.uniformBuffers.forward } },
      ],
    });
    this.bindGroups.softmax = this.device.createBindGroup({
      layout: this.pipelines.softmax.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.logitsBuffer } },
        { binding: 1, resource: { buffer: this.labelBuffer } },
        { binding: 2, resource: { buffer: this.probBuffer } },
        { binding: 3, resource: { buffer: this.gradLogitsBuffer } },
        { binding: 4, resource: { buffer: this.lossBuffer } },
        { binding: 5, resource: { buffer: this.uniformBuffers.softmax } },
      ],
    });
    this.bindGroups.gradWeights = this.device.createBindGroup({
      layout: this.pipelines.gradWeights.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.inputBuffer } },
        { binding: 1, resource: { buffer: this.gradLogitsBuffer } },
        { binding: 2, resource: { buffer: this.gradWeightBuffer } },
        { binding: 3, resource: { buffer: this.uniformBuffers.grad } },
      ],
    });
    this.bindGroups.reduceBias = this.device.createBindGroup({
      layout: this.pipelines.reduceBias.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradLogitsBuffer } },
        { binding: 1, resource: { buffer: this.gradBiasBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffers.reduce } },
      ],
    });
    this.bindGroups.scaleGradWeights = this.device.createBindGroup({
      layout: this.pipelines.scale.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradWeightBuffer } },
        { binding: 1, resource: { buffer: this.uniformBuffers.scaleWeights } },
      ],
    });
    this.bindGroups.scaleGradBias = this.device.createBindGroup({
      layout: this.pipelines.scale.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradBiasBuffer } },
        { binding: 1, resource: { buffer: this.uniformBuffers.scaleBias } },
      ],
    });
    this.bindGroups.sgdWeights = this.device.createBindGroup({
      layout: this.pipelines.sgd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.weightBuffer } },
        { binding: 1, resource: { buffer: this.gradWeightBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffers.sgdWeights } },
      ],
    });
    this.bindGroups.sgdBias = this.device.createBindGroup({
      layout: this.pipelines.sgd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.biasBuffer } },
        { binding: 1, resource: { buffer: this.gradBiasBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffers.sgdBias } },
      ],
    });
    this.bindGroups.adamWeights = this.device.createBindGroup({
      layout: this.pipelines.adam.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.weightBuffer } },
        { binding: 1, resource: { buffer: this.gradWeightBuffer } },
        { binding: 2, resource: { buffer: this.mWeightBuffer } },
        { binding: 3, resource: { buffer: this.vWeightBuffer } },
        { binding: 4, resource: { buffer: this.uniformBuffers.adamWeights } },
      ],
    });
    this.bindGroups.adamBias = this.device.createBindGroup({
      layout: this.pipelines.adam.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.biasBuffer } },
        { binding: 1, resource: { buffer: this.gradBiasBuffer } },
        { binding: 2, resource: { buffer: this.mBiasBuffer } },
        { binding: 3, resource: { buffer: this.vBiasBuffer } },
        { binding: 4, resource: { buffer: this.uniformBuffers.adamBias } },
      ],
    });
    this.bindGroups.accuracy = this.device.createBindGroup({
      layout: this.pipelines.accuracy.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.probBuffer } },
        { binding: 1, resource: { buffer: this.labelBuffer } },
        { binding: 2, resource: { buffer: this.accuracyMaskBuffer } },
        { binding: 3, resource: { buffer: this.uniformBuffers.accuracy } },
      ],
    });
  }

  dispose() {
    const bufferKeys = [
      'weightBuffer',
      'biasBuffer',
      'gradWeightBuffer',
      'gradBiasBuffer',
      'inputBuffer',
      'labelBuffer',
      'logitsBuffer',
      'probBuffer',
      'gradLogitsBuffer',
      'lossBuffer',
      'accuracyMaskBuffer',
      'mWeightBuffer',
      'vWeightBuffer',
      'mBiasBuffer',
      'vBiasBuffer',
    ];

    bufferKeys.forEach((key) => {
      const buffer = this[key];
      if (buffer?.destroy) {
        try {
          buffer.destroy();
        } catch (err) {
          console.warn(`Failed to destroy buffer ${buffer.label ?? key}`, err);
        }
      }
      this[key] = null;
    });

    Object.entries(this.uniformBuffers ?? {}).forEach(([name, buffer]) => {
      if (buffer?.destroy) {
        try {
          buffer.destroy();
        } catch (err) {
          console.warn(`Failed to destroy uniform buffer ${buffer.label ?? name}`, err);
        }
      }
    });

    this.uniformBuffers = {};
    this.bindGroups = {};
    this.initialized = false;
  }

  async trainBatch(batchImages, batchLabels, batchSize) {
    if (!this.initialized) {
      throw new Error('Model not initialized');
    }

    console.log(`Training batch of size ${batchSize}`);

    await writeBuffer(this.device, this.inputBuffer, batchImages);
    await writeBuffer(this.device, this.labelBuffer, batchLabels);

    await this.updateForwardUniforms(batchSize);
    await this.updateSoftmaxUniforms(batchSize);
    await this.updateReduceUniforms(batchSize);

    if (!this.bindGroups.forward) {
      this.createBindGroups();
    }

    // Capture GPU validation/internal issues so we can surface helpful errors.
    this.device.pushErrorScope('validation');
    this.device.pushErrorScope('internal');
    let scopesActive = true;
    const popGpuErrorScopes = async () => {
      if (!scopesActive) {
        return { internalError: null, validationError: null };
      }
      scopesActive = false;
      const internalError = await this.device.popErrorScope();
      const validationError = await this.device.popErrorScope();
      return { internalError, validationError };
    };

    try {
      const encoder = this.device.createCommandEncoder();

      // Forward pass.
      let pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.forward);
      pass.setBindGroup(0, this.bindGroups.forward);
      pass.dispatchWorkgroups(ceilDiv(batchSize * this.classes, WORKGROUP_128));
      pass.end();

      console.log('Completed forward pass');

      // Softmax + grad.
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.softmax);
      pass.setBindGroup(0, this.bindGroups.softmax);
      pass.dispatchWorkgroups(ceilDiv(batchSize, WORKGROUP_64));
      pass.end();

      console.log('Completed softmax and gradient computation');

      // Grad weights = input^T * gradLogits.
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.gradWeights);
      pass.setBindGroup(0, this.bindGroups.gradWeights);
      pass.dispatchWorkgroups(ceilDiv(this.features * this.classes, WORKGROUP_128));
      pass.end();

      console.log('Computed gradient for weights');

      // Reduce bias.
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.reduceBias);
      pass.setBindGroup(0, this.bindGroups.reduceBias);
      pass.dispatchWorkgroups(ceilDiv(this.classes, WORKGROUP_64));
      pass.end();

      console.log('Computed gradient for bias');

      // Scale gradients by 1/batch.
      await this.updateScaleUniform(this.uniformBuffers.scaleWeights, this.features * this.classes, 1 / batchSize);
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.scale);
      pass.setBindGroup(0, this.bindGroups.scaleGradWeights);
      pass.dispatchWorkgroups(ceilDiv(this.features * this.classes, WORKGROUP_128));
      pass.end();

      console.log('Scaled gradients for weights');

      await this.updateScaleUniform(this.uniformBuffers.scaleBias, this.classes, 1 / batchSize);
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.scale);
      pass.setBindGroup(0, this.bindGroups.scaleGradBias);
      pass.dispatchWorkgroups(ceilDiv(this.classes, WORKGROUP_128));
      pass.end();

      console.log('Scaled gradients for bias');

      // Optimizer update.
      if (this.optimizer === 'sgd') {
        await this.updateSgdUniform(this.uniformBuffers.sgdWeights, this.features * this.classes);
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.sgd);
        pass.setBindGroup(0, this.bindGroups.sgdWeights);
        pass.dispatchWorkgroups(ceilDiv(this.features * this.classes, WORKGROUP_128));
        pass.end();

        await this.updateSgdUniform(this.uniformBuffers.sgdBias, this.classes);
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.sgd);
        pass.setBindGroup(0, this.bindGroups.sgdBias);
        pass.dispatchWorkgroups(ceilDiv(this.classes, WORKGROUP_128));
        pass.end();
      } else {
        this.step += 1;
        this.beta1Power *= this.beta1;
        this.beta2Power *= this.beta2;
        await this.updateAdamUniform(this.uniformBuffers.adamWeights, this.features * this.classes);
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.adam);
        pass.setBindGroup(0, this.bindGroups.adamWeights);
        pass.dispatchWorkgroups(ceilDiv(this.features * this.classes, WORKGROUP_128));
        pass.end();

        await this.updateAdamUniform(this.uniformBuffers.adamBias, this.classes);
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.adam);
        pass.setBindGroup(0, this.bindGroups.adamBias);
        pass.dispatchWorkgroups(ceilDiv(this.classes, WORKGROUP_128));
        pass.end();
      }

      console.log('Updated weights and bias using optimizer');

      // Accuracy mask.
      await this.updateSoftmaxUniforms(batchSize);
      await writeBuffer(this.device, this.uniformBuffers.accuracy, new Uint32Array([batchSize, this.classes, 0]));
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.accuracy);
      pass.setBindGroup(0, this.bindGroups.accuracy);
      pass.dispatchWorkgroups(ceilDiv(batchSize, WORKGROUP_64));
      pass.end();

      console.log('Completed accuracy computation');

      const commandBuffer = encoder.finish();
      this.device.queue.submit([commandBuffer]);

      console.log('Submitted command buffer');

    } catch (err) {
      const { internalError, validationError } = await popGpuErrorScopes();
      const parts = [];
      if (err?.message) parts.push(err.message);
      if (internalError) parts.push(`internal: ${internalError.message}`);
      if (validationError) parts.push(`validation: ${validationError.message}`);
      const message = parts.length ? `trainBatch GPU execution failed: ${parts.join(' | ')}` : 'trainBatch GPU execution failed';
      console.error(message, err, internalError, validationError);
      throw new Error(message);
    }

    const { internalError, validationError } = await popGpuErrorScopes();
    if (internalError || validationError) {
      const parts = [];
      if (internalError) parts.push(`internal: ${internalError.message}`);
      if (validationError) parts.push(`validation: ${validationError.message}`);
      const message = `trainBatch GPU execution reported errors${parts.length ? `: ${parts.join(' | ')}` : ''}`;
      console.error(message, internalError, validationError);
      throw new Error(message);
    }

    const losses = await readBufferToArray(this.device, this.lossBuffer, Float32Array, batchSize);
    const accuracyMask = await readBufferToArray(this.device, this.accuracyMaskBuffer, Uint32Array, batchSize);

    let loss = 0;
    for (let i = 0; i < batchSize; i += 1) {
      loss += losses[i];
    }
    loss /= batchSize;

    let correct = 0;
    for (let i = 0; i < batchSize; i += 1) {
      correct += accuracyMask[i];
    }
    const accuracy = correct / batchSize;

    return { loss, accuracy };
  }

  async evaluateDataset(dataset, batchSize, fetchBatch) {
    let totalLoss = 0;
    let totalCorrect = 0;
    let totalSamples = 0;

    const batches = Math.ceil(dataset / batchSize);
    for (let i = 0; i < batches; i += 1) {
      const { images, labels, size } = fetchBatch(batchSize, i);
      const metrics = await this.trainBatchEvaluation(images, labels, size);
      totalLoss += metrics.loss * size;
      totalCorrect += metrics.accuracy * size;
      totalSamples += size;
    }

    return { loss: totalLoss / totalSamples, accuracy: totalCorrect / totalSamples };
  }

  async trainBatchEvaluation(batchImages, batchLabels, batchSize) {
    // Run forward only for evaluation.
    await writeBuffer(this.device, this.inputBuffer, batchImages);
    await writeBuffer(this.device, this.labelBuffer, batchLabels);

    await this.updateForwardUniforms(batchSize);
    await this.updateSoftmaxUniforms(batchSize);

    if (!this.bindGroups.forward) {
      this.createBindGroups();
    }

    const encoder = this.device.createCommandEncoder();

    let pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.forward);
    pass.setBindGroup(0, this.bindGroups.forward);
    pass.dispatchWorkgroups(ceilDiv(batchSize * this.classes, WORKGROUP_128));
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.softmax);
    pass.setBindGroup(0, this.bindGroups.softmax);
    pass.dispatchWorkgroups(ceilDiv(batchSize, WORKGROUP_64));
    pass.end();

    await writeBuffer(this.device, this.uniformBuffers.accuracy, new Uint32Array([batchSize, this.classes, 0]));
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.accuracy);
    pass.setBindGroup(0, this.bindGroups.accuracy);
    pass.dispatchWorkgroups(ceilDiv(batchSize, WORKGROUP_64));
    pass.end();

    this.device.queue.submit([encoder.finish()]);

    const losses = await readBufferToArray(this.device, this.lossBuffer, Float32Array, batchSize);
    const accuracyMask = await readBufferToArray(this.device, this.accuracyMaskBuffer, Uint32Array, batchSize);

    let loss = 0;
    for (let i = 0; i < batchSize; i += 1) {
      loss += losses[i];
    }
    loss /= batchSize;

    let correct = 0;
    for (let i = 0; i < batchSize; i += 1) {
      correct += accuracyMask[i];
    }
    const accuracy = correct / batchSize;
    return { loss, accuracy };
  }

  async predict(vector) {
    const batchSize = 1;
    const labels = new Float32Array(this.classes);
    await this.trainBatchEvaluation(vector, labels, batchSize);
    const probs = await readBufferToArray(this.device, this.probBuffer, Float32Array, this.classes);
    return probs.slice(0, this.classes);
  }

  async export() {
    const weights = await readBufferToArray(this.device, this.weightBuffer, Float32Array, this.features * this.classes);
    const bias = await readBufferToArray(this.device, this.biasBuffer, Float32Array, this.classes);
    return {
      optimizer: this.optimizer,
      learningRate: this.learningRate,
      weights: Array.from(weights),
      bias: Array.from(bias),
    };
  }

  async import(state) {
    if (!this.initialized) {
      throw new Error('Model must be initialized before importing');
    }
    if (state.weights?.length !== this.features * this.classes) {
      throw new Error('Invalid weight array');
    }
    if (state.bias?.length !== this.classes) {
      throw new Error('Invalid bias array');
    }
    this.optimizer = state.optimizer ?? this.optimizer;
    this.learningRate = state.learningRate ?? this.learningRate;
    await writeBuffer(this.device, this.weightBuffer, new Float32Array(state.weights));
    await writeBuffer(this.device, this.biasBuffer, new Float32Array(state.bias));
  }

  async getVisualization() {
    const weights = await readBufferToArray(this.device, this.weightBuffer, Float32Array, this.features * this.classes);
    return { weights };
  }
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
