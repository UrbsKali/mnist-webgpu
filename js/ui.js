const charts = {};

const drawState = {
  drawing: false,
  penSize: 14,
  autoPredict: true,
};

function ensureCharts() {
  if (charts.loss) {
    return;
  }
  const lossCtx = document.getElementById('chart-loss').getContext('2d');
  const trainAccCtx = document.getElementById('chart-train-acc').getContext('2d');
  const testAccCtx = document.getElementById('chart-test-acc').getContext('2d');
  const probsCtx = document.getElementById('chart-probs').getContext('2d');

  charts.loss = new Chart(lossCtx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Loss (per batch)', data: [], borderColor: '#10b981', tension: 0.2 }] },
    options: { scales: { x: { display: false } }, animation: false }
  });

  charts.trainAcc = new Chart(trainAccCtx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Train Accuracy', data: [], borderColor: '#2563eb', tension: 0.2 }] },
    options: {
      scales: { y: { suggestedMin: 0, suggestedMax: 1, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` } } },
      animation: false,
    }
  });

  charts.testAcc = new Chart(testAccCtx, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Test Accuracy', data: [], backgroundColor: '#f59e0b' }] },
    options: {
      scales: {
        y: { suggestedMin: 0, suggestedMax: 1, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` } },
      },
      animation: false,
    }
  });

  charts.probs = new Chart(probsCtx, {
    type: 'bar',
    data: {
      labels: Array.from({ length: 10 }, (_, i) => `${i}`),
      datasets: [{ label: 'Probability', data: Array(10).fill(0), backgroundColor: '#10b981' }],
    },
    options: {
      scales: {
        y: { suggestedMin: 0, suggestedMax: 1, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` } },
      },
      animation: false,
    }
  });
}

function setupDrawingCanvas(callbacks) {
  const drawCanvas = document.getElementById('draw-canvas');
  const ctx = drawCanvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);

  const previewCanvas = document.getElementById('preview-canvas');
  const previewCtx = previewCanvas.getContext('2d');
  const previewScratch = document.createElement('canvas');
  previewScratch.width = 28;
  previewScratch.height = 28;
  const previewScratchCtx = previewScratch.getContext('2d');

  const updatePreview = () => {
    const scale = drawCanvas.width / 28;
    const imageData = ctx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
  const previewImage = previewScratchCtx.createImageData(28, 28);
    for (let y = 0; y < 28; y += 1) {
      for (let x = 0; x < 28; x += 1) {
        let sum = 0;
        for (let dy = 0; dy < scale; dy += 1) {
          for (let dx = 0; dx < scale; dx += 1) {
            const sx = x * scale + dx;
            const sy = y * scale + dy;
            const idx = (sy * drawCanvas.width + sx) * 4;
            sum += imageData.data[idx];
          }
        }
        const avg = sum / (scale * scale);
        const offset = (y * 28 + x) * 4;
        previewImage.data[offset + 0] = avg;
        previewImage.data[offset + 1] = avg;
        previewImage.data[offset + 2] = avg;
        previewImage.data[offset + 3] = 255;
      }
    }
  previewScratchCtx.putImageData(previewImage, 0, 0);
  previewCtx.save();
  previewCtx.imageSmoothingEnabled = false;
  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewCtx.drawImage(previewScratch, 0, 0, previewCanvas.width, previewCanvas.height);
  previewCtx.restore();
    callbacks.onCanvasChange?.();
  };

  const drawPoint = (x, y) => {
    ctx.fillStyle = '#000000';
    ctx.beginPath();
    ctx.arc(x, y, drawState.penSize / 2, 0, Math.PI * 2);
    ctx.fill();
    updatePreview();
  };

  const pointerDown = (event) => {
    drawState.drawing = true;
    drawPoint(event.offsetX, event.offsetY);
  };

  const pointerMove = (event) => {
    if (!drawState.drawing) return;
    drawPoint(event.offsetX, event.offsetY);
  };

  const pointerUp = () => {
    drawState.drawing = false;
  };

  drawCanvas.addEventListener('pointerdown', pointerDown);
  drawCanvas.addEventListener('pointermove', pointerMove);
  drawCanvas.addEventListener('pointerup', pointerUp);
  drawCanvas.addEventListener('pointerleave', pointerUp);

  document.getElementById('pen-size').addEventListener('input', (event) => {
    drawState.penSize = Number(event.target.value);
  });

  document.getElementById('clear-draw').addEventListener('click', () => {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    updatePreview();
  });

  document.getElementById('invert-draw').addEventListener('click', () => {
    const data = ctx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
    for (let i = 0; i < data.data.length; i += 4) {
      const v = 255 - data.data[i];
      data.data[i + 0] = v;
      data.data[i + 1] = v;
      data.data[i + 2] = v;
    }
    ctx.putImageData(data, 0, 0);
    updatePreview();
  });

  document.getElementById('threshold-draw').addEventListener('click', () => {
    const data = ctx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
    for (let i = 0; i < data.data.length; i += 4) {
      const v = data.data[i] > 127 ? 255 : 0;
      data.data[i + 0] = v;
      data.data[i + 1] = v;
      data.data[i + 2] = v;
    }
    ctx.putImageData(data, 0, 0);
    updatePreview();
  });

  document.getElementById('predict').addEventListener('click', () => callbacks.onPredict?.());

  updatePreview();

  return {
    drawCanvas,
    previewCanvas,
    getImageVector() {
      const data = previewCtx.getImageData(0, 0, 28, 28);
      const vector = new Float32Array(28 * 28);
      for (let i = 0; i < vector.length; i += 1) {
        vector[i] = data.data[i * 4] / 255;
      }
      return vector;
    },
    refreshPreview: updatePreview,
  };
}

function updateBootProgress(progress, label) {
  const bar = document.getElementById('boot-progress');
  const status = document.getElementById('boot-status');
  bar.style.width = `${Math.max(0, Math.min(progress, 1)) * 100}%`;
  status.textContent = label;
}

function hideBootOverlay() {
  const overlay = document.getElementById('boot-overlay');
  overlay.classList.add('opacity-0');
  overlay.classList.add('pointer-events-none');
  setTimeout(() => overlay.classList.add('hidden'), 350);
}

function showWebGPURequired() {
  document.getElementById('webgpu-required').classList.remove('hidden');
}

function hideWebGPURequired() {
  document.getElementById('webgpu-required').classList.add('hidden');
}

function drawHeatmap(canvas, values, width, height, { min = -1, max = 1 } = {}) {
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(width, height);
  const range = max - min || 1;
  for (let i = 0; i < values.length; i += 1) {
    const v = (values[i] - min) / range;
    const color = heatColor(v);
    image.data[i * 4 + 0] = color[0];
    image.data[i * 4 + 1] = color[1];
    image.data[i * 4 + 2] = color[2];
    image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.save();
  ctx.scale(canvas.width / width, canvas.height / height);
  ctx.drawImage(canvas, 0, 0);
  ctx.restore();
}

function heatColor(value) {
  const v = Math.max(0, Math.min(1, value));
  const r = Math.floor(255 * Math.max(0, Math.min(1, (v - 0.5) * 2)));
  const b = Math.floor(255 * Math.max(0, Math.min(1, (0.5 - v) * 2)));
  const g = 255 - Math.abs(r - b);
  return [r, g, b];
}

function setWeightVisualizations(modelKey, payload) {
  const container = document.getElementById('weights-view');
  container.innerHTML = '';

  if (modelKey === 'lr' && payload?.weights) {
    const perClass = payload.weights.length / 10;
    for (let i = 0; i < 10; i += 1) {
      const canvas = document.createElement('canvas');
      canvas.width = 64;
      canvas.height = 64;
      canvas.className = 'border border-slate-200 rounded-md';
      const values = payload.weights.slice(i * perClass, (i + 1) * perClass);
      const min = Math.min(...values);
      const max = Math.max(...values);
      drawHeatmap(canvas, values, 28, 28, { min, max });
      const wrapper = document.createElement('div');
      wrapper.className = 'flex flex-col items-center text-xs gap-1';
      const label = document.createElement('span');
      label.textContent = `Class ${i}`;
      wrapper.appendChild(canvas);
      wrapper.appendChild(label);
      container.appendChild(wrapper);
    }
  }

  if (modelKey === 'cnn' && payload?.filters) {
    payload.filters.forEach((filter, index) => {
      const canvas = document.createElement('canvas');
      canvas.width = 64;
      canvas.height = 64;
      canvas.className = 'border border-slate-200 rounded-md';
      const min = Math.min(...filter);
      const max = Math.max(...filter);
      drawHeatmap(canvas, filter, 5, 5, { min, max });
      const wrapper = document.createElement('div');
      wrapper.className = 'flex flex-col items-center text-xs gap-1';
      const label = document.createElement('span');
      label.textContent = `Filter ${index}`;
      wrapper.appendChild(canvas);
      wrapper.appendChild(label);
      container.appendChild(wrapper);
    });

    if (payload.activations) {
      payload.activations.forEach((map, index) => {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        canvas.className = 'border border-slate-200 rounded-md';
        const min = Math.min(...map);
        const max = Math.max(...map);
        drawHeatmap(canvas, map, payload.activationWidth, payload.activationHeight, { min, max });
        const wrapper = document.createElement('div');
        wrapper.className = 'flex flex-col items-center text-xs gap-1';
        const label = document.createElement('span');
        label.textContent = `Activation ${index}`;
        wrapper.appendChild(canvas);
        wrapper.appendChild(label);
        container.appendChild(wrapper);
      });
    }
  }
}

function getDatasetConfig() {
  return {
    trainSamples: Number(document.getElementById('train-samples').value),
    testSamples: Number(document.getElementById('test-samples').value),
  };
}

function getHyperParams() {
  return {
    epochs: Number(document.getElementById('epochs').value),
    batchSize: Number(document.getElementById('batch-size').value),
    learningRate: Number(document.getElementById('learning-rate').value),
    optimizer: document.getElementById('optimizer').value,
    seed: Number(document.getElementById('seed').value),
  };
}

export function initUI(callbacks) {
  ensureCharts();
  hideWebGPURequired();
  const canvasBridge = setupDrawingCanvas(callbacks);

  document.getElementById('model-select').addEventListener('change', (event) => {
    callbacks.onModelChange?.(event.target.value);
  });

  document.getElementById('load-dataset').addEventListener('click', () => callbacks.onLoadDataset?.(getDatasetConfig()));
  document.getElementById('init-model').addEventListener('click', () => callbacks.onInitModel?.(getHyperParams()));
  document.getElementById('start-training').addEventListener('click', () => callbacks.onStartTraining?.());
  document.getElementById('pause-training').addEventListener('click', () => callbacks.onPauseTraining?.());
  document.getElementById('resume-training').addEventListener('click', () => callbacks.onResumeTraining?.());
  document.getElementById('reset-model').addEventListener('click', () => callbacks.onResetModel?.());
  document.getElementById('quick-train').addEventListener('click', () => callbacks.onQuickTrain?.());
  document.getElementById('evaluate').addEventListener('click', () => callbacks.onEvaluate?.());
  document.getElementById('save-indexeddb').addEventListener('click', () => callbacks.onSaveIndexedDb?.());
  document.getElementById('load-indexeddb').addEventListener('click', () => callbacks.onLoadIndexedDb?.());
  document.getElementById('download-model').addEventListener('click', () => callbacks.onDownloadModel?.());
  document.getElementById('upload-model').addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    if (file) {
      callbacks.onUploadModel?.(file);
    }
    event.target.value = '';
  });

  return {
    getDatasetConfig,
    getHyperParams,
    setBootProgress: updateBootProgress,
    hideBootOverlay,
    showWebGPURequired,
    hideWebGPURequired,
    setStatus(text) {
      document.getElementById('status-line').textContent = text;
    },
    setEpoch(value) {
      document.getElementById('status-epoch').textContent = value;
    },
    setBatch(value) {
      document.getElementById('status-batch').textContent = value;
    },
    setLoss(value) {
      document.getElementById('status-loss').textContent = value.toFixed(4);
    },
    setAccuracy(value) {
      document.getElementById('status-acc').textContent = `${(value * 100).toFixed(2)}%`;
    },
    setThroughput(value) {
      document.getElementById('status-throughput').textContent = value.toFixed(1);
    },
    addLossPoint(batch, loss) {
      charts.loss.data.labels.push(batch);
      charts.loss.data.datasets[0].data.push(loss);
      if (charts.loss.data.labels.length > 200) {
        charts.loss.data.labels.shift();
        charts.loss.data.datasets[0].data.shift();
      }
      charts.loss.update('none');
    },
    addTrainAccuracy(epoch, acc) {
      charts.trainAcc.data.labels.push(epoch);
      charts.trainAcc.data.datasets[0].data.push(acc);
      if (charts.trainAcc.data.labels.length > 50) {
        charts.trainAcc.data.labels.shift();
        charts.trainAcc.data.datasets[0].data.shift();
      }
      charts.trainAcc.update('none');
    },
    addTestAccuracy(label, acc) {
      charts.testAcc.data.labels.push(label);
      charts.testAcc.data.datasets[0].data.push(acc);
      if (charts.testAcc.data.labels.length > 20) {
        charts.testAcc.data.labels.shift();
        charts.testAcc.data.datasets[0].data.shift();
      }
      charts.testAcc.update('none');
    },
    updateProbabilities(probs) {
      const values = Array.from(probs);
      charts.probs.data.datasets[0].data = values;
      charts.probs.update('none');
      const top = values
        .map((value, index) => ({ value, index }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 3);
      document.getElementById('top-predictions').textContent = top
        .map(({ value, index }) => `${index}: ${(value * 100).toFixed(1)}%`)
        .join(' \u2022 ');
    },
    renderWeights(modelKey, payload) {
      setWeightVisualizations(modelKey, payload);
    },
    getCanvasVector() {
      return canvasBridge.getImageVector();
    },
    refreshPreview: canvasBridge.refreshPreview,
  };
}
