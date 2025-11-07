# mnist-example-webgpu

Mnist but WebGPU for my CreaTech master at IFT

## Shader Debug Harness

The repository now includes a standalone shader test harness at `debug.html`. It compiles every compute shader in the project, runs a tiny synthetic workload, and compares the GPU output against CPU-calculated expectations so you can debug shader math in isolation.

### Running the tests

1. Serve the repository root with any static file server (for example, `npx http-server .` or the VS Code Live Server extension).
2. Open `http://localhost:PORT/debug.html` in a WebGPU-enabled browser (Chrome 113+ with WebGPU enabled).
3. Click **Run All Tests** or execute shaders individually to inspect their outputs and error messages.

Each row reports the shader path, a short description, live status, and mismatches (if any). GPU validation errors surfaced through error scopes are shown directly in the table for faster iteration.
