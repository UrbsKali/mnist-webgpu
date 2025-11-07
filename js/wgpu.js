const shaderCache = new Map();
const LOG_BUFFERS = (() => {
  if (typeof window === 'undefined') {
    return false;
  }
  try {
    return window.localStorage?.getItem('webgpu-log-buffers') === '1';
  } catch (err) {
    return false;
  }
})();

export function logBugger(event, details) {
  if (!LOG_BUFFERS) {
    return;
  }
  try {
    const stamp = new Date().toISOString();
  console.debug(`[WebGPU][${stamp}] ${event}`, details);
  } catch (err) {
    console.debug(`[WebGPU] ${event}`);
  }
}

export async function initWebGPU() {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported');
  }

  // const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
  if (!adapter) {
    throw new Error('Failed to acquire GPU adapter');
  }

  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
  });

  const queue = device.queue;
  attachDeviceErrorHandlers(device);

  return { adapter, device, queue };
}

function attachDeviceErrorHandlers(device) {
  if (typeof device?.addEventListener === 'function') {
    device.addEventListener('uncapturederror', (event) => {
      const error = event.error;
      logBugger('device.uncapturederror', {
        message: error?.message ?? 'Unknown GPU error',
        type: error?.constructor?.name ?? 'GPUError',
        stack: error?.stack,
      });
    });
  }

  if (device?.lost instanceof Promise) {
    device.lost
      .then((info) => {
        if (!info) {
          return;
        }
        logBugger('device.lost', {
          reason: info.reason,
          message: info.message,
        });
      })
      .catch((err) => {
        logBugger('device.lost.handlerError', {
          message: err?.message ?? String(err),
        });
      });
  }
}

export async function loadShaderModule(device, url) {
  if (shaderCache.has(url)) {
    return shaderCache.get(url);
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load shader ${url}`);
  }
  const code = await response.text();
  const module = device.createShaderModule({ code, label: url });
  shaderCache.set(url, module);
  return module;
}

function getArrayView(arrayLike) {
  if (arrayLike instanceof ArrayBuffer) {
    return new Uint8Array(arrayLike);
  }
  if (ArrayBuffer.isView(arrayLike)) {
    return new Uint8Array(arrayLike.buffer, arrayLike.byteOffset, arrayLike.byteLength);
  }
  throw new Error('Unsupported buffer data type');
}

export async function createBuffer(device, array, usage, label) {
  const view = getArrayView(array);
  const byteLength = view.byteLength;
  const size = align(byteLength, 4);
  const finalUsage = usage | GPUBufferUsage.COPY_DST;
  const buffer = device.createBuffer({
    label,
    size,
    usage: finalUsage,
  });

  await writeBuffer(device, buffer, view);

  logBugger('createBuffer', {
    label: buffer.label ?? label ?? '(unnamed)',
    alignedSize: size,
    originalSize: byteLength,
    usage: finalUsage,
  });
  return buffer;
}

export function createEmptyBuffer(device, byteLength, usage, label) {
  const size = align(byteLength, 4);
  const buffer = device.createBuffer({
    label,
    size,
    usage,
  });
  logBugger('createEmptyBuffer', {
    label: buffer.label ?? label ?? '(unnamed)',
    size,
    requestedSize: byteLength,
    usage,
  });
  return buffer;
}

export async function writeBuffer(device, buffer, data, offset = 0) {
  const view = getArrayView(data);
  const alignedSize = align(view.byteLength, 4);
  const label = buffer.label ?? '(unnamed)';

  logBugger('writeBuffer.begin', {
    label,
    byteLength: view.byteLength,
    alignedSize,
    offset,
  });

  const copySize = Math.min(alignedSize, Math.max(0, (buffer?.size ?? alignedSize) - offset));
  if (copySize <= 0) {
    logBugger('writeBuffer.skip', {
      label,
      reason: 'copySize <= 0',
      requestedSize: alignedSize,
      offset,
      bufferSize: buffer?.size ?? null,
    });
    return;
  }

  const copyLength = Math.min(view.byteLength, copySize);
  if (copyLength < view.byteLength) {
    logBugger('writeBuffer.truncate', {
      label,
      providedBytes: view.byteLength,
      copyLength,
      copySize,
      bufferSize: buffer?.size ?? null,
      offset,
    });
    throw new RangeError('writeBuffer: data length exceeds target buffer capacity');
  }

  const attemptDirectMap = async () => {
    const hasMapAsync = typeof buffer?.mapAsync === 'function';
    const usage = typeof buffer?.usage === 'number' ? buffer.usage : 0;
    const canMapWrite = (usage & GPUBufferUsage.MAP_WRITE) !== 0;

    if (!hasMapAsync || !canMapWrite) {
      logBugger('writeBuffer.directMap.skip', {
        label,
        hasMapAsync,
        usage,
        canMapWrite,
      });
      return false;
    }
    try {
      await buffer.mapAsync(GPUMapMode.WRITE, offset, copySize);
      const mapped = buffer.getMappedRange(offset, copySize);
      const target = new Uint8Array(mapped);
      if (copySize > copyLength) {
        target.fill(0);
      }
      target.set(view.subarray(0, copyLength));
      buffer.unmap();
      logBugger('writeBuffer.directMap', {
        label,
        byteLength: view.byteLength,
        copySize,
        offset,
      });
      return true;
    } catch (err) {
      logBugger('writeBuffer.directMap.fallback', {
        label,
        message: err?.message ?? String(err),
      });
      return false;
    }
  };

  if (await attemptDirectMap()) {
    return;
  }

  const stagingBuffer = device.createBuffer({
    label: `${label}-staging`,
    size: copySize,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
  });

  try {
    await stagingBuffer.mapAsync(GPUMapMode.WRITE);
  } catch (err) {
    stagingBuffer.destroy();
    logBugger('writeBuffer.staging.mapError', {
      label,
      message: err?.message ?? String(err),
    });
    throw err;
  }

  const mapped = stagingBuffer.getMappedRange();
  const target = new Uint8Array(mapped);
  if (copySize > copyLength) {
    target.fill(0);
  }
  target.set(view.subarray(0, copyLength));
  stagingBuffer.unmap();

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(stagingBuffer, 0, buffer, offset, copySize);
  device.queue.submit([encoder.finish()]);

  const completion = device.queue
    .onSubmittedWorkDone()
    .then(() => {
      stagingBuffer.destroy();
      logBugger('writeBuffer.complete', {
        label,
        byteLength: view.byteLength,
        copySize,
        offset,
        path: 'staging-copy',
      });
    })
    .catch((err) => {
      logBugger('writeBuffer.complete.error', {
        label,
        message: err?.message ?? String(err),
      });
      stagingBuffer.destroy();
    });

  // Avoid awaiting completion so callers don't block on GPU work; attach suppression to
  // keep unhandled rejection warnings away if nobody observes the promise.
  completion.catch(() => {});
}

export async function readBufferToArray(device, buffer, constructor, length) {
  const size = align(constructor.BYTES_PER_ELEMENT * length, 4);
  logBugger('readBuffer.begin', {
    label: buffer.label ?? '(unnamed)',
    byteLength: size,
    elementType: constructor.name,
    elementCount: length,
  });
  const readBuffer = device.createBuffer({
    label: `${buffer.label ?? 'temp'}-readback`,
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, readBuffer.size);
  device.queue.submit([commandEncoder.finish()]);

  await device.queue.onSubmittedWorkDone();

  try {
    await readBuffer.mapAsync(GPUMapMode.READ);
  } catch (err) {
    logBugger('readBuffer.error', {
      label: buffer.label ?? '(unnamed)',
      byteLength: size,
      elementType: constructor.name,
      elementCount: length,
      message: err?.message ?? String(err),
    });
    readBuffer.destroy();
    throw err;
  }
  const copyArray = new constructor(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  readBuffer.destroy();
  logBugger('readBuffer.complete', {
    label: buffer.label ?? '(unnamed)',
    byteLength: copyArray.byteLength ?? size,
    elementCount: copyArray.length,
    elementType: constructor.name,
  });
  return copyArray;
}

export async function createComputePipeline(device, descriptor) {
  const label = descriptor?.label ?? '(unnamed)';
  device.pushErrorScope('validation');
  device.pushErrorScope('internal');

  let pipeline;
  let thrownError = null;
  try {
    pipeline = device.createComputePipeline(descriptor);
  } catch (err) {
    thrownError = err;
  }

  const internalError = await device.popErrorScope();
  const validationError = await device.popErrorScope();
  const scopeError = validationError ?? internalError;

  if (scopeError) {
    logBugger('pipeline.error', {
      label,
      type: scopeError.constructor?.name ?? 'GPUError',
      message: scopeError.message ?? String(scopeError),
      stack: scopeError.stack,
    });
  }

  if (thrownError) {
    logBugger('pipeline.exception', {
      label,
      message: thrownError?.message ?? String(thrownError),
      stack: thrownError?.stack,
    });
    throw thrownError;
  }

  if (scopeError) {
    throw scopeError;
  }

  logBugger('pipeline.created', {
    label,
    entryPoint: descriptor?.compute?.entryPoint,
  });

  return pipeline;
}

export function createBindGroup(device, layout, entries, label) {
  return device.createBindGroup({
    label,
    layout,
    entries,
  });
}

export function align(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

export function ceilDiv(a, b) {
  return Math.floor((a + b - 1) / b);
}

export function createTimer(device) {
  if (!device.features.has('timestamp-query')) {
    return {
      enabled: false,
      async measure(callback) {
        const start = performance.now();
        await callback();
        const end = performance.now();
        return end - start;
      },
    };
  }

  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
  const resolveBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  return {
    enabled: true,
    async measure(callback) {
      const encoder = device.createCommandEncoder();
      encoder.writeTimestamp(querySet, 0);
      await callback(encoder);
      encoder.writeTimestamp(querySet, 1);
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      device.queue.submit([encoder.finish()]);
      await resolveBuffer.mapAsync(GPUMapMode.READ);
      const timestamps = new BigUint64Array(resolveBuffer.getMappedRange());
      const elapsed = Number((timestamps[1] - timestamps[0]) * BigInt(device.limits.timestampPeriod ?? 1)) / 1_000_000;
      resolveBuffer.unmap();
      return elapsed;
    },
  };
}
