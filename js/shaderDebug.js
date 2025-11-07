import {
  initWebGPU,
  loadShaderModule,
  createComputePipeline,
  readBufferToArray,
  ceilDiv,
  writeBuffer,
} from './wgpu.js';
import { shaderTests, DEFAULT_FLOAT_TOLERANCE } from './shaderTests.js';
import { LogisticRegressionModel } from './models/lrModel.js';
import { DataConstants } from './data.js';

const runAllButton = document.getElementById('run-all');
const deviceStatusLabel = document.getElementById('device-status');
const tableBody = document.getElementById('shader-test-body');
const lrRunButton = document.getElementById('lr-run');
const lrStageSelect = document.getElementById('lr-step-select');
const lrOptimizerSelect = document.getElementById('lr-optimizer-select');
const lrSummaryLabel = document.getElementById('lr-debug-summary');
const lrStageResults = document.getElementById('lr-stage-results');

const rowMap = new Map();
let gpuContext = null;
let deviceInitPromise = null;

function formatNumber(value) {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  if (Number.isInteger(value)) {
    return value.toString();
  }
  const absValue = Math.abs(value);
  if (absValue !== 0 && (absValue >= 1000 || absValue < 1e-3)) {
    return value.toExponential(2);
  }
  return value.toFixed(4).replace(/\.0+$/, '').replace(/0+$/, '').replace(/\.$/, '');
}

function formatArray(values, limit = 6) {
  const preview = values.slice(0, limit).map(formatNumber);
  const suffix = values.length > limit ? ', …' : '';
  return `[${preview.join(', ')}${suffix}]`;
}

function pushGpuErrorScopes(device) {
  device.pushErrorScope('validation');
  device.pushErrorScope('internal');
  let active = true;
  return async () => {
    if (!active) {
      return { internalError: null, validationError: null };
    }
    active = false;
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    return { internalError, validationError };
  };
}

function normalizeDispatch(dispatch) {
  if (dispatch == null) {
    return { x: 1, y: 1, z: 1 };
  }
  if (typeof dispatch === 'number') {
    const value = Math.max(1, dispatch);
    return { x: value, y: 1, z: 1 };
  }
  if (Array.isArray(dispatch)) {
    return {
      x: Math.max(1, dispatch[0] ?? 1),
      y: Math.max(1, dispatch[1] ?? 1),
      z: Math.max(1, dispatch[2] ?? 1),
    };
  }
  return {
    x: Math.max(1, dispatch.x ?? 1),
    y: Math.max(1, dispatch.y ?? 1),
    z: Math.max(1, dispatch.z ?? 1),
  };
}

function defaultVerify(outputs) {
  if (!outputs?.length) {
    return { pass: true, message: 'No outputs were requested for verification.' };
  }
  const notes = [];
  for (const output of outputs) {
    if (!output.expected) {
      continue;
    }
    const actual = Array.from(output.data);
    const expected = Array.from(output.expected);
    if (actual.length !== expected.length) {
      return {
        pass: false,
        message: `${output.name}: length mismatch (expected ${expected.length}, received ${actual.length})`,
      };
    }
    const tolerance = output.tolerance ?? DEFAULT_FLOAT_TOLERANCE;
    for (let i = 0; i < actual.length; i += 1) {
      const diff = Math.abs(actual[i] - expected[i]);
      if (Number.isNaN(actual[i]) || diff > tolerance) {
        return {
          pass: false,
          message: `${output.name}: mismatch at index ${i} · expected ${formatNumber(expected[i])}, got ${formatNumber(actual[i])} (tolerance ${tolerance})`,
        };
      }
    }
    notes.push(`${output.name} ✓ ${formatArray(actual)}`);
  }
  return { pass: true, message: notes.join(' | ') };
}

async function ensureDevice() {
  if (gpuContext) {
    return gpuContext.device;
  }
  if (!deviceInitPromise) {
    deviceStatusLabel.textContent = 'Requesting WebGPU adapter…';
    deviceInitPromise = initWebGPU()
      .then((context) => {
        gpuContext = context;
        const adapterName = context.adapter?.name ?? 'Unknown adapter';
        deviceStatusLabel.textContent = `WebGPU ready · ${adapterName}`;
        return context.device;
      })
      .catch((error) => {
        deviceStatusLabel.textContent = `Failed to initialize WebGPU: ${error?.message ?? error}`;
        throw error;
      })
      .finally(() => {
        deviceInitPromise = null;
      });
  }
  return deviceInitPromise;
}

function createStatusChip(text, variant) {
  const chip = document.createElement('span');
  chip.className = `status-chip status-${variant}`;
  chip.textContent = text;
  return chip;
}

function setRowStatus(rowInfo, variant, message) {
  const { statusCell, detailsCell } = rowInfo;
  const statusMap = {
    pending: { text: 'Pending', className: 'status-pending' },
    running: { text: 'Running…', className: 'status-running' },
    pass: { text: 'Passed', className: 'status-pass' },
    fail: { text: 'Failed', className: 'status-fail' },
  };
  const statusConfig = statusMap[variant] ?? statusMap.pending;
  statusCell.innerHTML = '';
  const chip = createStatusChip(statusConfig.text, variant);
  statusCell.appendChild(chip);
  detailsCell.textContent = message ?? '';
}

async function executeTest(device, test) {
  const module = await loadShaderModule(device, test.shader);
  const pipeline = await createComputePipeline(device, {
    label: `debug-${test.id}`,
    layout: 'auto',
    compute: { module, entryPoint: test.entryPoint ?? 'main' },
  });

  const popErrors = pushGpuErrorScopes(device);
  let context;
  const outputs = [];
  try {
    context = await test.setup({ device, pipeline });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    const bindGroups = context.bindGroups ?? [];
    bindGroups.forEach((group, index) => {
      pass.setBindGroup(index, group);
    });
    const dispatch = normalizeDispatch(context.dispatch);
    pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
    pass.end();
    device.queue.submit([encoder.finish()]);

    if (context.outputs) {
      for (const output of context.outputs) {
        const data = await readBufferToArray(device, output.buffer, output.ctor, output.length);
        outputs.push({ ...output, data });
      }
    }

    const { internalError, validationError } = await popErrors();
    if (internalError || validationError) {
      const messages = [];
      if (internalError) {
        messages.push(`internal: ${internalError.message}`);
      }
      if (validationError) {
        messages.push(`validation: ${validationError.message}`);
      }
      return {
        pass: false,
        message: `GPU error scopes reported problems (${messages.join(' | ')})`,
        outputs,
      };
    }

    const verification = test.verify ? await test.verify(outputs, context) : defaultVerify(outputs);
    return {
      pass: Boolean(verification?.pass),
      message: verification?.message ?? (verification?.pass ? 'Outputs matched expected values.' : 'Verification failed.'),
      outputs,
    };
  } catch (error) {
    const { internalError, validationError } = await popErrors();
    const parts = [error?.message ?? String(error)];
    if (internalError) {
      parts.push(`internal: ${internalError.message}`);
    }
    if (validationError) {
      parts.push(`validation: ${validationError.message}`);
    }
    return {
      pass: false,
      message: parts.join(' | '),
      outputs,
    };
  } finally {
    try {
      context?.destroy?.();
    } catch (destroyError) {
      console.warn('Failed to release test resources', destroyError);
    }
  }
}

async function runSingleTest(rowInfo) {
  const { test, runButton } = rowInfo;
  try {
    runButton.disabled = true;
    setRowStatus(rowInfo, 'running', 'Dispatching compute pass…');
    const device = await ensureDevice();
    const result = await executeTest(device, test);
    const variant = result.pass ? 'pass' : 'fail';
    setRowStatus(rowInfo, variant, result.message);
  } catch (error) {
    setRowStatus(rowInfo, 'fail', error?.message ?? String(error));
  } finally {
    runButton.disabled = false;
  }
}

async function runAllTests() {
  runAllButton.disabled = true;
  try {
    const device = await ensureDevice();
    if (!device) {
      return;
    }
    for (const test of shaderTests) {
      const rowInfo = rowMap.get(test.id);
      // eslint-disable-next-line no-await-in-loop
      await runSingleTest(rowInfo);
    }
  } finally {
    runAllButton.disabled = false;
  }
}

function createRow(test) {
  const tr = document.createElement('tr');

  const shaderCell = document.createElement('td');
  const shaderTitle = document.createElement('strong');
  shaderTitle.textContent = test.label;
  const shaderSubtitle = document.createElement('div');
  shaderSubtitle.className = 'summary';
  shaderSubtitle.textContent = test.shader;
  shaderCell.appendChild(shaderTitle);
  shaderCell.appendChild(shaderSubtitle);

  const descriptionCell = document.createElement('td');
  descriptionCell.textContent = test.description;

  const statusCell = document.createElement('td');
  const initialChip = createStatusChip('Pending', 'pending');
  statusCell.appendChild(initialChip);

  const detailsCell = document.createElement('td');
  detailsCell.className = 'details';
  detailsCell.textContent = 'Not run yet.';

  const actionsCell = document.createElement('td');
  actionsCell.className = 'actions';
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = 'Run';
  button.addEventListener('click', () => {
    if (!button.disabled) {
      runSingleTest(rowInfo).catch((err) => {
        console.error('Shader test failed', err);
      });
    }
  });
  actionsCell.appendChild(button);

  tr.appendChild(shaderCell);
  tr.appendChild(descriptionCell);
  tr.appendChild(statusCell);
  tr.appendChild(detailsCell);
  tr.appendChild(actionsCell);

  const rowInfo = {
    test,
    tr,
    statusCell,
    detailsCell,
    runButton: button,
  };
  rowMap.set(test.id, rowInfo);
  return tr;
}

function bootstrapTable() {
  shaderTests.forEach((test) => {
    const row = createRow(test);
    tableBody.appendChild(row);
  });
}

const LR_WORKGROUP_128 = 128;
const LR_WORKGROUP_64 = 64;
const LR_DEBUG_BATCH_SIZE = 8;
const LR_DEBUG_LEARNING_RATE = 0.001;
const LR_DEBUG_SEED = 7;
const LR_SAMPLE_LIMIT = 12;

let lrDebugModel = null;

function buildDebugBatch(batchSize, features, classes) {
  const images = new Float32Array(batchSize * features);
  const labels = new Float32Array(batchSize * classes);
  for (let batch = 0; batch < batchSize; batch += 1) {
    const imageOffset = batch * features;
    for (let idx = 0; idx < features; idx += 1) {
      images[imageOffset + idx] = ((batch * 13 + idx) % 23) / 22;
    }
    const target = batch % classes;
    labels[batch * classes + target] = 1;
  }
  return { images, labels };
}

async function prepareDebugModel(device, optimizer) {
  if (!lrDebugModel) {
    lrDebugModel = new LogisticRegressionModel(device);
    await lrDebugModel.compile();
  }
  if (lrDebugModel.initialized) {
    lrDebugModel.dispose();
  }
  await lrDebugModel.initialize({
    batchSize: LR_DEBUG_BATCH_SIZE,
    learningRate: LR_DEBUG_LEARNING_RATE,
    optimizer,
    seed: LR_DEBUG_SEED,
  });
  lrDebugModel.createBindGroups();
  return lrDebugModel;
}

async function dispatchComputePass(device, label, pipeline, bindGroup, dispatch) {
  const popErrors = pushGpuErrorScopes(device);
  let thrown = null;
  try {
    const encoder = device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
    pass.end();
    device.queue.submit([encoder.finish()]);
  } catch (error) {
    thrown = error;
  }
  const { internalError, validationError } = await popErrors();
  if (thrown || internalError || validationError) {
    const parts = [];
    if (thrown) {
      parts.push(thrown.message ?? String(thrown));
    }
    if (internalError) {
      parts.push(`internal: ${internalError.message}`);
    }
    if (validationError) {
      parts.push(`validation: ${validationError.message}`);
    }
    const message = parts.length ? parts.join(' | ') : 'GPU dispatch failed';
    return { success: false, message };
  }
  return {
    success: true,
    message: `Dispatched ${dispatch.x}×${dispatch.y}×${dispatch.z} workgroups`,
  };
}

async function runStage(ctx, config) {
  try {
    if (config.before) {
      // eslint-disable-next-line no-await-in-loop
      await config.before(ctx);
    }
  } catch (error) {
    return { success: false, message: error?.message ?? String(error), outputs: [] };
  }

  const dispatch = normalizeDispatch(config.dispatch);
  const result = await dispatchComputePass(ctx.device, config.label, config.pipeline, config.bindGroup, dispatch);
  const outputs = [];
  if (result.success && config.reads) {
    const readList = config.reads(ctx) ?? [];
    for (const read of readList) {
      // eslint-disable-next-line no-await-in-loop
      const typed = await readBufferToArray(
        ctx.device,
        read.buffer,
        read.ctor ?? Float32Array,
        read.length,
      );
      const limit = read.sampleLimit ?? LR_SAMPLE_LIMIT;
      outputs.push({
        label: `${read.label} [0:${read.length}]`,
        text: formatArray(Array.from(typed), limit),
      });
    }
  }
  return { success: result.success, message: result.message, outputs };
}

const LR_STAGES = [
  {
    id: 'forward',
    label: 'Forward pass · matmul_bias',
    async execute(ctx) {
      const result = await runStage(ctx, {
        label: 'lr-debug-forward',
        pipeline: ctx.model.pipelines.forward,
        bindGroup: ctx.model.bindGroups.forward,
        dispatch: { x: ceilDiv(ctx.batchSize * ctx.model.classes, LR_WORKGROUP_128) },
        reads: () => [
          {
            label: 'logits',
            buffer: ctx.model.logitsBuffer,
            ctor: Float32Array,
            length: Math.min(ctx.batchSize * ctx.model.classes, 16),
            sampleLimit: 8,
          },
        ],
      });
      if (result.success) {
        result.message = 'Computed logits (A·W + b).';
      }
      return result;
    },
  },
  {
    id: 'softmax',
    label: 'Softmax + loss',
    async execute(ctx) {
      const result = await runStage(ctx, {
        label: 'lr-debug-softmax',
        pipeline: ctx.model.pipelines.softmax,
        bindGroup: ctx.model.bindGroups.softmax,
        before: async () => {
          await ctx.model.updateSoftmaxUniforms(ctx.batchSize);
        },
        dispatch: { x: ceilDiv(ctx.batchSize, LR_WORKGROUP_64) },
        reads: () => [
          {
            label: 'probabilities',
            buffer: ctx.model.probBuffer,
            ctor: Float32Array,
            length: Math.min(ctx.batchSize * ctx.model.classes, 16),
            sampleLimit: 8,
          },
          {
            label: 'gradLogits',
            buffer: ctx.model.gradLogitsBuffer,
            ctor: Float32Array,
            length: Math.min(ctx.batchSize * ctx.model.classes, 16),
            sampleLimit: 8,
          },
          {
            label: 'losses',
            buffer: ctx.model.lossBuffer,
            ctor: Float32Array,
            length: Math.min(ctx.batchSize, 8),
            sampleLimit: 8,
          },
        ],
      });
      if (result.success) {
        result.message = 'Softmax probabilities and per-sample losses ready.';
      }
      return result;
    },
  },
  {
    id: 'gradWeights',
    label: 'Gradient weights · matmul_at_b',
    async execute(ctx) {
      const result = await runStage(ctx, {
        label: 'lr-debug-gradW',
        pipeline: ctx.model.pipelines.gradWeights,
        bindGroup: ctx.model.bindGroups.gradWeights,
        dispatch: { x: ceilDiv(ctx.model.features * ctx.model.classes, LR_WORKGROUP_128) },
        reads: () => [
          {
            label: 'gradWeights',
            buffer: ctx.model.gradWeightBuffer,
            ctor: Float32Array,
            length: Math.min(ctx.model.features * ctx.model.classes, 16),
            sampleLimit: 8,
          },
        ],
      });
      if (result.success) {
        result.message = 'Accumulated weight gradients (Xᵗ·∂L/∂logits).';
      }
      return result;
    },
  },
  {
    id: 'reduceBias',
    label: 'Gradient bias · reduce_sum_axis0',
    async execute(ctx) {
      const result = await runStage(ctx, {
        label: 'lr-debug-gradB',
        pipeline: ctx.model.pipelines.reduceBias,
        bindGroup: ctx.model.bindGroups.reduceBias,
        before: async () => {
          await ctx.model.updateReduceUniforms(ctx.batchSize);
        },
        dispatch: { x: ceilDiv(ctx.model.classes, LR_WORKGROUP_64) },
        reads: () => [
          {
            label: 'gradBias',
            buffer: ctx.model.gradBiasBuffer,
            ctor: Float32Array,
            length: Math.min(ctx.model.classes, 10),
            sampleLimit: 10,
          },
        ],
      });
      if (result.success) {
        result.message = 'Summed bias gradient across the batch.';
      }
      return result;
    },
  },
  {
    id: 'scaleWeights',
    label: 'Scale weight gradients',
    async execute(ctx) {
      const elements = ctx.model.features * ctx.model.classes;
      const result = await runStage(ctx, {
        label: 'lr-debug-scaleW',
        pipeline: ctx.model.pipelines.scale,
        bindGroup: ctx.model.bindGroups.scaleGradWeights,
        before: async () => {
          await ctx.model.updateScaleUniform(ctx.model.uniformBuffers.scaleWeights, elements, 1 / ctx.batchSize);
        },
        dispatch: { x: ceilDiv(elements, LR_WORKGROUP_128) },
        reads: () => [
          {
            label: 'scaledGradWeights',
            buffer: ctx.model.gradWeightBuffer,
            ctor: Float32Array,
            length: Math.min(elements, 16),
            sampleLimit: 8,
          },
        ],
      });
      if (result.success) {
        result.message = 'Scaled weight gradients by 1/batchSize.';
      }
      return result;
    },
  },
  {
    id: 'scaleBias',
    label: 'Scale bias gradients',
    async execute(ctx) {
      const result = await runStage(ctx, {
        label: 'lr-debug-scaleB',
        pipeline: ctx.model.pipelines.scale,
        bindGroup: ctx.model.bindGroups.scaleGradBias,
        before: async () => {
          await ctx.model.updateScaleUniform(ctx.model.uniformBuffers.scaleBias, ctx.model.classes, 1 / ctx.batchSize);
        },
        dispatch: { x: ceilDiv(ctx.model.classes, LR_WORKGROUP_128) },
        reads: () => [
          {
            label: 'scaledGradBias',
            buffer: ctx.model.gradBiasBuffer,
            ctor: Float32Array,
            length: Math.min(ctx.model.classes, 10),
            sampleLimit: 10,
          },
        ],
      });
      if (result.success) {
        result.message = 'Scaled bias gradients by 1/batchSize.';
      }
      return result;
    },
  },
  {
    id: 'optimizer',
    label: 'Optimizer update',
    async execute(ctx) {
      const weightSize = ctx.model.features * ctx.model.classes;
      const biasSize = ctx.model.classes;
      if (ctx.optimizer === 'sgd') {
        await ctx.model.updateSgdUniform(ctx.model.uniformBuffers.sgdWeights, weightSize);
        const resWeights = await dispatchComputePass(
          ctx.device,
          'lr-debug-sgd-weights',
          ctx.model.pipelines.sgd,
          ctx.model.bindGroups.sgdWeights,
          normalizeDispatch({ x: ceilDiv(weightSize, LR_WORKGROUP_128) }),
        );
        if (!resWeights.success) {
          return { success: false, message: `Weights · ${resWeights.message}`, outputs: [] };
        }
        await ctx.model.updateSgdUniform(ctx.model.uniformBuffers.sgdBias, biasSize);
        const resBias = await dispatchComputePass(
          ctx.device,
          'lr-debug-sgd-bias',
          ctx.model.pipelines.sgd,
          ctx.model.bindGroups.sgdBias,
          normalizeDispatch({ x: ceilDiv(biasSize, LR_WORKGROUP_128) }),
        );
        if (!resBias.success) {
          return { success: false, message: `Bias · ${resBias.message}`, outputs: [] };
        }
        const outputs = [];
        const weightSlice = await readBufferToArray(
          ctx.device,
          ctx.model.weightBuffer,
          Float32Array,
          Math.min(weightSize, 16),
        );
        outputs.push({
          label: 'weights',
          text: formatArray(Array.from(weightSlice), 8),
        });
        const biasSlice = await readBufferToArray(
          ctx.device,
          ctx.model.biasBuffer,
          Float32Array,
          Math.min(biasSize, 10),
        );
        outputs.push({
          label: 'bias',
          text: formatArray(Array.from(biasSlice), 10),
        });
        return {
          success: true,
          message: 'Applied SGD parameter update.',
          outputs,
        };
      }

      ctx.model.step += 1;
      ctx.model.beta1Power *= ctx.model.beta1;
      ctx.model.beta2Power *= ctx.model.beta2;

      await ctx.model.updateAdamUniform(ctx.model.uniformBuffers.adamWeights, weightSize);
      const resWeights = await dispatchComputePass(
        ctx.device,
        'lr-debug-adam-weights',
        ctx.model.pipelines.adam,
        ctx.model.bindGroups.adamWeights,
        normalizeDispatch({ x: ceilDiv(weightSize, LR_WORKGROUP_128) }),
      );
      if (!resWeights.success) {
        return { success: false, message: `Weights · ${resWeights.message}`, outputs: [] };
      }

      await ctx.model.updateAdamUniform(ctx.model.uniformBuffers.adamBias, biasSize);
      const resBias = await dispatchComputePass(
        ctx.device,
        'lr-debug-adam-bias',
        ctx.model.pipelines.adam,
        ctx.model.bindGroups.adamBias,
        normalizeDispatch({ x: ceilDiv(biasSize, LR_WORKGROUP_128) }),
      );
      if (!resBias.success) {
        return { success: false, message: `Bias · ${resBias.message}`, outputs: [] };
      }

      const outputs = [];
      const weightSlice = await readBufferToArray(
        ctx.device,
        ctx.model.weightBuffer,
        Float32Array,
        Math.min(weightSize, 16),
      );
      outputs.push({
        label: 'weights',
        text: formatArray(Array.from(weightSlice), 8),
      });
      const biasSlice = await readBufferToArray(
        ctx.device,
        ctx.model.biasBuffer,
        Float32Array,
        Math.min(biasSize, 10),
      );
      outputs.push({
        label: 'bias',
        text: formatArray(Array.from(biasSlice), 10),
      });
      const mSlice = await readBufferToArray(
        ctx.device,
        ctx.model.mWeightBuffer,
        Float32Array,
        Math.min(weightSize, 16),
      );
      outputs.push({
        label: 'm (1st moment)',
        text: formatArray(Array.from(mSlice), 8),
      });
      const vSlice = await readBufferToArray(
        ctx.device,
        ctx.model.vWeightBuffer,
        Float32Array,
        Math.min(weightSize, 16),
      );
      outputs.push({
        label: 'v (2nd moment)',
        text: formatArray(Array.from(vSlice), 8),
      });
      return {
        success: true,
        message: `Applied Adam update (step ${ctx.model.step}).`,
        outputs,
      };
    },
  },
  {
    id: 'accuracy',
    label: 'Accuracy mask',
    async execute(ctx) {
      const result = await runStage(ctx, {
        label: 'lr-debug-accuracy',
        pipeline: ctx.model.pipelines.accuracy,
        bindGroup: ctx.model.bindGroups.accuracy,
        before: async () => {
          await ctx.model.updateSoftmaxUniforms(ctx.batchSize);
          const info = new Uint32Array([ctx.batchSize, ctx.model.classes, 0]);
          await writeBuffer(ctx.device, ctx.model.uniformBuffers.accuracy, info);
        },
        dispatch: { x: ceilDiv(ctx.batchSize, LR_WORKGROUP_64) },
        reads: () => [
          {
            label: 'accuracyMask',
            buffer: ctx.model.accuracyMaskBuffer,
            ctor: Uint32Array,
            length: Math.min(ctx.batchSize, 16),
            sampleLimit: 16,
          },
        ],
      });
      if (result.success) {
        result.message = 'Derived per-sample accuracy mask.';
      }
      return result;
    },
  },
];

function populateStageSelect() {
  if (!lrStageSelect) {
    return;
  }
  lrStageSelect.innerHTML = '';
  LR_STAGES.forEach((stage) => {
    const option = document.createElement('option');
    option.value = stage.id;
    option.textContent = stage.label;
    lrStageSelect.appendChild(option);
  });
}

function renderLrResults(results) {
  if (!lrStageResults) {
    return;
  }
  lrStageResults.innerHTML = '';
  results.forEach((entry) => {
    const card = document.createElement('div');
    card.className = `stage-card ${entry.success ? 'pass' : 'fail'}`;
    const title = document.createElement('h3');
    title.textContent = entry.label;
    const message = document.createElement('div');
    message.className = 'message';
    message.textContent = entry.message ?? '';
    card.appendChild(title);
    card.appendChild(message);
    if (entry.outputs?.length) {
      const list = document.createElement('ul');
      entry.outputs.forEach((output) => {
        const item = document.createElement('li');
        item.textContent = `${output.label}: ${output.text}`;
        list.appendChild(item);
      });
      card.appendChild(list);
    }
    lrStageResults.appendChild(card);
  });
}

async function runLrDebugSequence(device, upToStageId, optimizer) {
  const stageIndex = LR_STAGES.findIndex((stage) => stage.id === upToStageId);
  if (stageIndex === -1) {
    throw new Error(`Unknown stage id "${upToStageId}"`);
  }

  const model = await prepareDebugModel(device, optimizer);
  const batchSize = LR_DEBUG_BATCH_SIZE;
  const { images, labels } = buildDebugBatch(batchSize, DataConstants.imageSize, DataConstants.numClasses);

  await writeBuffer(device, model.inputBuffer, images);
  await writeBuffer(device, model.labelBuffer, labels);

  await model.updateForwardUniforms(batchSize);
  await model.updateSoftmaxUniforms(batchSize);
  await model.updateReduceUniforms(batchSize);

  const ctx = { device, model, batchSize, optimizer };
  const results = [];
  for (let idx = 0; idx <= stageIndex; idx += 1) {
    const stage = LR_STAGES[idx];
    // eslint-disable-next-line no-await-in-loop
    const outcome = await stage.execute(ctx);
    results.push({ id: stage.id, label: stage.label, ...outcome });
    if (!outcome.success) {
      break;
    }
  }
  return results;
}

async function handleRunLrDebug() {
  if (!lrRunButton || !lrStageSelect || !lrOptimizerSelect) {
    return;
  }
  lrRunButton.disabled = true;
  if (lrStageResults) {
    lrStageResults.innerHTML = '';
  }
  if (lrSummaryLabel) {
    lrSummaryLabel.textContent = 'Preparing debug run…';
  }
  try {
    const device = await ensureDevice();
    if (!device) {
      throw new Error('WebGPU device unavailable');
    }
    const results = await runLrDebugSequence(device, lrStageSelect.value, lrOptimizerSelect.value);
    renderLrResults(results);
    if (lrSummaryLabel) {
      if (!results.length) {
        lrSummaryLabel.textContent = 'No stages executed.';
      } else {
        const finalStage = results[results.length - 1];
        lrSummaryLabel.textContent = finalStage.success
          ? `Completed through "${finalStage.label}".`
          : `Failed during "${finalStage.label}".`;
      }
    }
  } catch (error) {
    if (lrSummaryLabel) {
      lrSummaryLabel.textContent = `Debug run failed: ${error?.message ?? String(error)}`;
    }
    if (lrStageResults) {
      lrStageResults.innerHTML = '';
    }
    console.error('Logistic regression debug run failed', error);
  } finally {
    lrRunButton.disabled = false;
  }
}

function initLrDebugger() {
  if (!lrRunButton || !lrStageSelect || !lrOptimizerSelect || !lrStageResults || !lrSummaryLabel) {
    return;
  }
  populateStageSelect();
  lrStageSelect.value = LR_STAGES[LR_STAGES.length - 1].id;
  lrSummaryLabel.textContent = 'Select a step and run to inspect intermediate buffers.';
  lrRunButton.addEventListener('click', () => {
    handleRunLrDebug().catch((error) => {
      console.error('Failed to execute LR debug run', error);
    });
  });
}

if (runAllButton) {
  runAllButton.addEventListener('click', () => {
    runAllTests().catch((error) => {
      console.error('Failed to run shader tests', error);
    });
  });
}

if (tableBody) {
  bootstrapTable();
}

initLrDebugger();
