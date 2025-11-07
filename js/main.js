import { initWebGPU } from './wgpu.js';
import { MNISTData, DataConstants } from './data.js';
import { initUI } from './ui.js';
import { saveModelToIndexedDB, loadModelFromIndexedDB, downloadModelJSON, readModelFile } from './persist.js';
import { LogisticRegressionModel } from './models/lrModel.js';
import { CnnModel } from './models/cnnModel.js';

const state = {
  device: null,
  queue: null,
  data: new MNISTData(),
  dataLoaded: false,
  datasetConfig: { trainSamples: 2000, testSamples: 500 },
  models: {},
  currentModelKey: 'lr',
  hyperparams: {},
  batchScratch: {
    trainImages: null,
    trainLabels: null,
    testImages: null,
    testLabels: null,
  },
  training: {
    running: false,
    paused: false,
    cancelRequested: false,
    epoch: 0,
    totalEpochs: 0,
    batchCounter: 0,
  },
  ui: null,
  inferenceBusy: false,
  predictionScheduled: false,
};

let cleanupHandlersRegistered = false;
let cleanupExecuted = false;

const callbacks = {
  onModelChange: handleModelChange,
  onLoadDataset: handleLoadDataset,
  onInitModel: handleInitModel,
  onStartTraining: () => handleStartTraining(),
  onPauseTraining: handlePauseTraining,
  onResumeTraining: handleResumeTraining,
  onResetModel: handleResetModel,
  onQuickTrain: handleQuickTrain,
  onEvaluate: handleEvaluate,
  onSaveIndexedDb: handleSaveIndexedDb,
  onLoadIndexedDb: handleLoadIndexedDb,
  onDownloadModel: handleDownloadModel,
  onUploadModel: handleUploadModel,
  onCanvasChange: () => schedulePrediction(true),
  onPredict: () => schedulePrediction(false),
};

async function bootstrap() {
  if (!navigator.gpu) {
    document.getElementById('webgpu-required').classList.remove('hidden');
    return;
  }

  state.ui = initUI(callbacks);
  state.ui.setStatus('Booting…');

  try {
    const { device, queue } = await initWebGPU();
    state.device = device;
    console.log('WebGPU initialized:', device);
    state.queue = queue;
    state.models.lr = new LogisticRegressionModel(device);
    state.models.cnn = new CnnModel(device);
  registerLifecycleHandlers();

    await loadDataset(state.datasetConfig, true);
    await Promise.all([state.models.lr.compile(), state.models.cnn.compile()]);

    state.ui.setStatus('Ready. Initialize a model to begin.');
    state.ui.hideBootOverlay();
    schedulePrediction(false);
  } catch (err) {
    console.error(err);
    state.ui.showWebGPURequired();
    state.ui.setStatus('Unable to initialize WebGPU.');
  }
}

async function loadDataset(config, showProgress = false) {
  state.datasetConfig = { ...config };
  if (showProgress) {
    state.ui.setBootProgress(0.01, 'Preparing dataset fetch');
  } else {
    state.ui.setStatus('Loading dataset…');
  }
  await state.data.load(config, (progress, label) => {
    if (showProgress) {
      state.ui.setBootProgress(progress, label);
    } else {
      state.ui.setStatus(label);
    }
  });
  state.dataLoaded = true;
  state.ui.setStatus(`Dataset loaded (train ${state.data.trainSize}, test ${state.data.testSize})`);
}

function ensureScratchBuffers(batchSize) {
  const imageElements = batchSize * DataConstants.imageSize;
  const labelElements = batchSize * DataConstants.numClasses;
  if (!state.batchScratch.trainImages || state.batchScratch.trainImages.length !== imageElements) {
    state.batchScratch.trainImages = new Float32Array(imageElements);
    state.batchScratch.trainLabels = new Float32Array(labelElements);
  }
  if (!state.batchScratch.testImages || state.batchScratch.testImages.length !== imageElements) {
    state.batchScratch.testImages = new Float32Array(imageElements);
    state.batchScratch.testLabels = new Float32Array(labelElements);
  }
}

function getCurrentModel() {
  return state.models[state.currentModelKey];
}

function getCurrentHyperparams() {
  return state.hyperparams[state.currentModelKey];
}

async function handleModelChange(modelKey) {
  state.currentModelKey = modelKey;
  const model = getCurrentModel();
  if (model?.initialized) {
    const viz = await model.getVisualization();
    state.ui.renderWeights(modelKey, viz);
  } else {
    state.ui.renderWeights(modelKey, null);
  }
  schedulePrediction(false);
}

async function handleLoadDataset(config) {
  if (state.training.running) {
    state.ui.setStatus('Stop training before reloading the dataset.');
    return;
  }
  await loadDataset(config, false);
}

async function handleInitModel(hyper) {
  if (!state.device) {
    state.ui.setStatus('WebGPU device unavailable.');
    return;
  }
  if (!state.dataLoaded) {
    state.ui.setStatus('Load the dataset before initializing the model.');
    return;
  }

  const model = getCurrentModel();
  try {
    await model.initialize({
      batchSize: hyper.batchSize,
      learningRate: hyper.learningRate,
      optimizer: hyper.optimizer,
      seed: hyper.seed,
    });
    state.hyperparams[state.currentModelKey] = { ...hyper };
    ensureScratchBuffers(hyper.batchSize);
    resetTrainingState();

    const viz = await model.getVisualization();
    state.ui.renderWeights(state.currentModelKey, viz);
    state.ui.setStatus('Model initialized.');
    schedulePrediction(false);
  } catch (err) {
    console.error(err);
    state.ui.setStatus('Failed to initialize model. See console for details.');
  }
}

function resetTrainingState() {
  state.training.running = false;
  state.training.paused = false;
  state.training.cancelRequested = false;
  state.training.epoch = 0;
  state.training.totalEpochs = 0;
  state.training.batchCounter = 0;
  if (state.ui) {
    state.ui.setEpoch(0);
    state.ui.setBatch(0);
    state.ui.setLoss(0);
    state.ui.setAccuracy(0);
    state.ui.setThroughput(0);
  }
}

async function handleStartTraining(customEpochs) {
  const model = getCurrentModel();
  if (!model?.initialized) {
    state.ui.setStatus('Initialize the current model first.');
    return;
  }
  if (!state.dataLoaded) {
    state.ui.setStatus('Dataset not loaded.');
    return;
  }
  if (state.training.running) {
    state.ui.setStatus('Training already running.');
    return;
  }

  const uiHyper = state.ui.getHyperParams();
  const storedHyper = getCurrentHyperparams();
  if (!storedHyper) {
    state.ui.setStatus('Initialize the model before training.');
    return;
  }
  if (uiHyper.batchSize !== storedHyper.batchSize) {
    state.ui.setStatus('Batch size differs from initialized model. Re-initialize first.');
    return;
  }

  const totalEpochs = customEpochs ?? uiHyper.epochs;
  if (!customEpochs) {
    storedHyper.epochs = uiHyper.epochs;
  }
  storedHyper.learningRate = uiHyper.learningRate;
  storedHyper.optimizer = uiHyper.optimizer;
  model.learningRate = storedHyper.learningRate;
  if (model.optimizer !== storedHyper.optimizer) {
    model.optimizer = storedHyper.optimizer;
    if (storedHyper.optimizer === 'adam') {
      model.beta1Power = model.beta1;
      model.beta2Power = model.beta2;
      model.step = 0;
    }
  }

  state.training.running = true;
  state.training.paused = false;
  state.training.cancelRequested = false;
  state.training.totalEpochs = totalEpochs;
  state.training.batchCounter = 0;
  state.training.epoch = 0;
  state.ui.setStatus('Training started.');
  const batchesPerEpoch = Math.ceil(state.data.trainSize / storedHyper.batchSize);
  state.ui.setEpoch(`0/${totalEpochs}`);
  state.ui.setBatch(`0/${batchesPerEpoch}`);
  state.ui.setLoss(0);
  state.ui.setAccuracy(0);
  state.ui.setThroughput(0);
  runTrainingLoop(totalEpochs).catch((err) => {
    console.error(err);
    state.ui.setStatus('Training failed. See console.');
    resetTrainingState();
  });
}

async function runTrainingLoop(totalEpochs) {
  const model = getCurrentModel();
  const hyper = getCurrentHyperparams();
  const batchSize = hyper.batchSize;
  const batchesPerEpoch = Math.ceil(state.data.trainSize / batchSize);

  for (let epochIndex = 0; epochIndex < totalEpochs; epochIndex += 1) {
    state.training.epoch = epochIndex + 1;
    let epochLoss = 0;
    let epochAcc = 0;
    let epochSamples = 0;

    for (let batchIndex = 0; batchIndex < batchesPerEpoch; batchIndex += 1) {
      if (state.training.cancelRequested) {
        state.ui.setStatus('Training cancelled.');
        resetTrainingState();
        return;
      }
      while (state.training.paused && !state.training.cancelRequested) {
        // eslint-disable-next-line no-await-in-loop
        await sleep(80);
      }
      if (state.training.cancelRequested) {
        break;
      }

      const { images, labels, size } = state.data.getTrainBatch(
        batchSize,
        batchIndex,
        state.batchScratch.trainImages,
        state.batchScratch.trainLabels,
      );

      console.log(`Training epoch ${epochIndex + 1}, batch ${batchIndex + 1} / ${batchesPerEpoch} (size ${size})`);

      const t0 = performance.now();
      // eslint-disable-next-line no-await-in-loop
      const metrics = await model.trainBatch(images, labels, size);
      const dt = performance.now() - t0;

      epochLoss += metrics.loss * size;
      epochAcc += metrics.accuracy * size;
      epochSamples += size;
      state.training.batchCounter += 1;

      const throughput = size / (dt / 1000 || 1);
      state.ui.setEpoch(`${epochIndex + 1}/${totalEpochs}`);
      state.ui.setBatch(`${batchIndex + 1}/${batchesPerEpoch}`);
      state.ui.setLoss(metrics.loss);
      state.ui.setAccuracy(metrics.accuracy);
      state.ui.setThroughput(throughput);
      state.ui.addLossPoint(state.training.batchCounter, metrics.loss);

      // Allow the UI and event loop to breathe between batches.
      // eslint-disable-next-line no-await-in-loop
      await sleep(0);
    }

    const epochLossAvg = epochLoss / Math.max(epochSamples, 1);
    const epochAccAvg = epochAcc / Math.max(epochSamples, 1);
    state.ui.addTrainAccuracy(epochIndex + 1, epochAccAvg);
    state.ui.setStatus(
      `Epoch ${epochIndex + 1}/${totalEpochs} complete · Loss ${epochLossAvg.toFixed(4)} · Acc ${(epochAccAvg * 100).toFixed(2)}%`,
    );

    const viz = await model.getVisualization();
    state.ui.renderWeights(state.currentModelKey, viz);
    schedulePrediction(false);
  }

  state.training.running = false;
  state.training.paused = false;
  state.ui.setStatus('Training finished.');
}

function handlePauseTraining() {
  if (!state.training.running) {
    state.ui.setStatus('Training not active.');
    return;
  }
  state.training.paused = true;
  state.ui.setStatus('Training paused.');
}

function handleResumeTraining() {
  if (!state.training.running) {
    state.ui.setStatus('Training not active.');
    return;
  }
  state.training.paused = false;
  state.ui.setStatus('Resuming training…');
}

async function handleResetModel() {
  if (state.training.running) {
    state.training.cancelRequested = true;
    state.ui.setStatus('Stopping training before reset…');
    while (state.training.running) {
      // eslint-disable-next-line no-await-in-loop
      await sleep(80);
    }
  }
  const hyper = getCurrentHyperparams();
  if (!hyper) {
    state.ui.setStatus('Initialize the model first.');
    return;
  }
  await handleInitModel(hyper);
}

function handleQuickTrain() {
  const hyper = getCurrentHyperparams();
  if (!hyper) {
    state.ui.setStatus('Initialize the model before training.');
    return;
  }
  handleStartTraining(1);
}

async function handleEvaluate() {
  const model = getCurrentModel();
  const hyper = getCurrentHyperparams();
  if (!model?.initialized || !hyper) {
    state.ui.setStatus('Initialize the current model before evaluation.');
    return;
  }
  ensureScratchBuffers(hyper.batchSize);
  state.ui.setStatus('Evaluating on test set…');
  const metrics = await model.evaluateDataset(state.data.testSize, hyper.batchSize, (batchSize, batchIndex) =>
    state.data.getTestBatch(batchSize, batchIndex, state.batchScratch.testImages, state.batchScratch.testLabels),
  );
  const label = new Date().toLocaleTimeString();
  state.ui.addTestAccuracy(label, metrics.accuracy);
  state.ui.setStatus(`Test accuracy ${(metrics.accuracy * 100).toFixed(2)}% · Loss ${metrics.loss.toFixed(4)}`);
}

async function handleSaveIndexedDb() {
  const model = getCurrentModel();
  if (!model?.initialized) {
    state.ui.setStatus('Initialize a model before saving.');
    return;
  }
  try {
    const payload = await model.export();
    await saveModelToIndexedDB(state.currentModelKey, {
      modelKey: state.currentModelKey,
      payload,
      hyperparams: getCurrentHyperparams(),
      datasetConfig: state.datasetConfig,
      savedAt: Date.now(),
    });
    state.ui.setStatus('Model saved to IndexedDB.');
  } catch (err) {
    console.error(err);
    state.ui.setStatus('Failed to save model.');
  }
}

async function handleLoadIndexedDb() {
  const record = await loadModelFromIndexedDB(state.currentModelKey);
  if (!record) {
    state.ui.setStatus('No saved model for this configuration.');
    return;
  }
  const model = getCurrentModel();
  if (!model?.initialized) {
    state.ui.setStatus('Initialize the model before loading saved weights.');
    return;
  }
  try {
    await model.import(record.payload);
    if (record.hyperparams) {
      state.hyperparams[state.currentModelKey] = { ...record.hyperparams };
    }
    state.ui.setStatus('Model restored from IndexedDB.');
    const viz = await model.getVisualization();
    state.ui.renderWeights(state.currentModelKey, viz);
    schedulePrediction(false);
  } catch (err) {
    console.error(err);
    state.ui.setStatus('Failed to load model from IndexedDB.');
  }
}

async function handleDownloadModel() {
  const model = getCurrentModel();
  if (!model?.initialized) {
    state.ui.setStatus('Initialize the model before exporting.');
    return;
  }
  const payload = await model.export();
  const filename = `${state.currentModelKey}-mnist-${Date.now()}.json`;
  downloadModelJSON(filename, {
    modelKey: state.currentModelKey,
    payload,
    hyperparams: getCurrentHyperparams(),
    datasetConfig: state.datasetConfig,
  });
  state.ui.setStatus('Model exported as JSON.');
}

async function handleUploadModel(file) {
  const model = getCurrentModel();
  if (!model?.initialized) {
    state.ui.setStatus('Initialize the model before importing weights.');
    return;
  }
  try {
    const json = await readModelFile(file);
    if (json.modelKey && json.modelKey !== state.currentModelKey) {
      state.ui.setStatus(`Uploaded weights belong to ${json.modelKey}. Switch model first.`);
      return;
    }
    await model.import(json.payload);
    if (json.hyperparams) {
      state.hyperparams[state.currentModelKey] = { ...json.hyperparams };
    }
    state.ui.setStatus('Model weights imported.');
    const viz = await model.getVisualization();
    state.ui.renderWeights(state.currentModelKey, viz);
    schedulePrediction(false);
  } catch (err) {
    console.error(err);
    state.ui.setStatus('Failed to import model.');
  }
}

function schedulePrediction(fromCanvas) {
  if (state.inferenceBusy) {
    state.predictionScheduled = true;
    return;
  }
  if (fromCanvas && state.predictionScheduled) {
    return;
  }
  state.predictionScheduled = true;
  queueMicrotask(async () => {
    state.predictionScheduled = false;
    await runPrediction();
  });
}

async function runPrediction() {
  const model = getCurrentModel();
  if (!model?.initialized) {
    return;
  }
  state.inferenceBusy = true;
  try {
    const vector = state.ui.getCanvasVector();
    const probs = await model.predict(vector);
    state.ui.updateProbabilities(probs);
  } catch (err) {
    console.error(err);
  } finally {
    state.inferenceBusy = false;
    if (state.predictionScheduled) {
      schedulePrediction(false);
    }
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

window.addEventListener('DOMContentLoaded', bootstrap);

function cleanupWebGPU() {
  if (cleanupExecuted) {
    return;
  }
  cleanupExecuted = true;

  state.training.cancelRequested = true;
  state.training.running = false;
  state.training.paused = false;

  Object.values(state.models).forEach((model) => {
    if (typeof model?.dispose === 'function') {
      try {
        model.dispose();
      } catch (err) {
        console.warn('Failed to dispose model', err);
      }
    }
  });

  state.batchScratch.trainImages = null;
  state.batchScratch.trainLabels = null;
  state.batchScratch.testImages = null;
  state.batchScratch.testLabels = null;

  if (typeof state.device?.destroy === 'function') {
    try {
      state.device.destroy();
    } catch (err) {
      console.warn('Failed to destroy GPU device', err);
    }
  }

  state.device = null;
  state.queue = null;
}

function registerLifecycleHandlers() {
  if (cleanupHandlersRegistered) {
    return;
  }

  const handler = () => cleanupWebGPU();
  window.addEventListener('pagehide', handler);
  window.addEventListener('beforeunload', handler);
  cleanupHandlersRegistered = true;
}
