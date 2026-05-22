# PRISM — CLAUDE.md
## Parallel Runtime for ISA-agnostic Serial-to-GPU Migration

This file is the single source of truth for every architectural and
implementation decision in PRISM. Read it fully before writing any code.
If a decision is recorded here, do not override it silently — leave a
`# DEVIATION: <reason>` comment and open a GitHub issue.

Every decision below traces to an explicit user choice. Nothing is assumed.

---

## Decisions log

| # | Question | Decision |
|---|----------|----------|
| 1 | Project name | PRISM |
| 2 | GPU target | Any NVIDIA GPU with CUDA 12+ (auto-detect SM) |
| 3 | Input formats | `.bc` + `.ll` + source files (PRISM calls clang) |
| 4 | Offload strategy | Profile-guided (CPU first, profile, then JIT to GPU) |
| 5 | Unproven parallelism posture | User flag: `--aggressive` / `--conservative` |
| 6 | GPU offload candidates | Loops + linalg ops + embarrassingly parallel + reductions |
| 7 | Memory strategy | Unified memory (`cudaMallocManaged`) |
| 8 | GPU kernel failure | Silent CPU fallback + `WARN` log |
| 9 | Multi-GPU | Round-robin across all available CUDA devices |
| 10 | Exposed surfaces | C++ lib + CLI (`prism-run`) + Python bindings + C API |
| 11 | Test framework | lit (LLVM Integrated Tester) + GTest |
| 12 | Licence | MIT |
| 13 | Syscalls / I/O in loop body | Always CPU |
| 14 | Recursive functions | Always CPU |
| 15 | Target OS | Linux + Windows + macOS |
| 16 | C++ standard | C++17 (matches LLVM 18; "no preference" → use LLVM's own standard) |
| 17 | CI/CD | GitHub Actions |
| 18 | Documentation | Sphinx (user docs) + Doxygen (API docs) via Breathe extension |

---

## Project goal

PRISM is a zero-touch, profile-guided CPU-to-GPU VM/executor. Any program
compiled to LLVM bitcode (or any source file PRISM can compile via clang)
runs inside PRISM unmodified. PRISM:

1. Executes the program on CPU with lightweight profiling instrumentation.
2. Identifies hot loops and parallel regions from the profile.
3. Compiles those regions to CUDA kernels and re-executes them on GPU.
4. Manages all host-device memory automatically via unified memory.
5. Falls back to CPU silently for any region that cannot be offloaded.

The user does nothing beyond pointing PRISM at their program.

---

## Non-goals for v1 (do not implement)

- OpenCL, Vulkan, Metal, WebGPU backends.
- Source-level transformation (we operate on LLVM IR only internally).
- Automatic vectorisation (separate from parallelisation).
- Profiling-based inter-procedural optimisation.
- Ahead-of-time GPU compilation (everything is JIT in v1).
- Windows GPU support in CI (no self-hosted Windows+GPU runner available;
  GPU tests run on Linux CI only. Windows build must compile and pass
  CPU-only tests.)

---

## Tech stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Core language | C++17 | — |
| Compiler | Clang + LLD | 18+ |
| IR framework | LLVM + MLIR | 18+ |
| GPU codegen | MLIR NVVM dialect -> PTX -> CUDA Runtime | CUDA 12+ |
| CPU JIT | LLVM ORC JIT v2 | same as LLVM |
| Profiling | LLVM IR instrumentation (custom pass) | — |
| Build | CMake 3.25+ + Ninja | — |
| Testing | lit + GTest 1.14+ | — |
| Python bindings | nanobind 2+ | — |
| C API | Plain C header, implemented in C++ | — |
| Docs | Sphinx 7+ + Doxygen 1.9+ + Breathe 4+ | — |
| Formatting | clang-format (LLVM style) | — |
| Linting | clang-tidy | — |
| CI | GitHub Actions | — |

### LLVM build flags (required)

```cmake
-DLLVM_ENABLE_PROJECTS="mlir;clang;lld"
-DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AArch64"
-DMLIR_ENABLE_CUDA_RUNNER=ON
-DLLVM_ENABLE_ASSERTIONS=ON
-DCMAKE_BUILD_TYPE=RelWithDebInfo
-DLLVM_ENABLE_LLD=ON
```

Store the install prefix in `LLVM_DIR` (cmake) and `MLIR_DIR`. PRISM's
CMake finds them via `find_package(MLIR REQUIRED CONFIG)`.

---

## Input handling

PRISM accepts three input modes. All three ultimately produce an
`mlir::ModuleOp` in the LLVM dialect before entering the pipeline.

### Mode 1 — LLVM bitcode (`.bc`)

Parse with `llvm::getLazyBitcodeModule`, then import via
`mlir::translateLLVMIRToMLIR` (`mlir/Target/LLVMIR/Import.h`).

### Mode 2 — LLVM IR text (`.ll`)

Parse with `llvm::parseIRFile`, then same import path as Mode 1.

### Mode 3 — Source files (`.c`, `.cpp`, `.cc`, `.cxx`, `.rs`, any clang-supported)

PRISM invokes clang as a library (not a subprocess) using
`clang::CompilerInstance`. Flags used:

```
-O1 -emit-llvm -g0 -fno-discard-value-names
```

`-O1` (not `-O0`) is intentional: we want mem2reg and basic cleanup before
analysis. `-O2`+ is forbidden here because it may eliminate loops that PRISM
could parallelise.

The resulting `llvm::Module*` is then imported via Mode 1's path.

Auto-detect source language from file extension. If unknown, error out:
"PRISM: unrecognised source extension '<ext>'. Pass a .bc or .ll file,
or use a clang-supported source extension."

---

## Pipeline overview

```
Input (source / .ll / .bc)
        |
        v
  [1] Ingestion       -- produce mlir::ModuleOp in LLVM dialect
        |
        v
  [2] CPU execution   -- ORC JIT + profiling instrumentation
        |  (repeat until --profile-threshold invocations observed)
        v
  [3] Profile analysis -- identify hot regions, classify kind
        |
        v
  [4] MLIR lifting    -- LLVM dialect -> affine / linalg / scf / gpu
        |  (only for hot regions)
        v
  [5] Analysis        -- dependency graph, parallelism, cost model
        |
        v
  [6] Offload decision -- GPU or CPU per region
        |
        +--GPU--> [7a] GPU codegen  -> NVVM -> PTX -> cubin
        |
        +--CPU--> [7b] CPU codegen  -> ORC JIT (re-use existing)
        |
        v
  [8] Memory manager  -- unified memory allocation + prefetch hints
        |
        v
  [9] Execution engine -- multi-GPU round-robin, streams, fallback
```

---

## Stage 1 — Ingestion (`lib/Ingestion/`)

**Header:** `include/prism/Ingestion/Ingester.h`

```cpp
namespace prism {

enum class InputKind { Bitcode, LLVMIR, Source };

struct IngestOptions {
  InputKind kind;
  std::string path;
  std::vector<std::string> extraClangArgs; // Source mode only
};

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
ingest(mlir::MLIRContext& ctx, const IngestOptions& opts);

} // namespace prism
```

- Always run `mlir::verify()` after import. Return `llvm::Error` on failure.
- Never throw exceptions. Use `llvm::Expected<T>` throughout.
- Do not perform any transformation here.

---

## Stage 2 — CPU execution with profiling (`lib/Profiling/`)

**Header:** `include/prism/Profiling/Profiler.h`

On the first `--profile-threshold` invocations (default: 100), PRISM runs
the program entirely on CPU via ORC JIT v2 with profiling instrumentation
injected by a custom LLVM IR pass (`ProfilerInstrumentPass`).

### What the instrumentation measures (per loop)

The pass inserts calls to `__prism_loop_enter(loop_id)` and
`__prism_loop_exit(loop_id, trip_count, bytes_read, bytes_written, wall_ns)`
at each loop's entry and back-edge. The runtime implements these as
lock-free atomic updates to a `ProfileStore`.

```cpp
struct LoopProfile {
  uint64_t loopId;
  std::atomic<uint64_t> invocationCount{0};
  std::atomic<uint64_t> totalTripCount{0};
  std::atomic<uint64_t> totalBytesRead{0};
  std::atomic<uint64_t> totalBytesWritten{0};
  std::atomic<uint64_t> totalWallNs{0};
};

class ProfileStore {
public:
  void record(uint64_t loopId, uint64_t tripCount,
              uint64_t bytesRead, uint64_t bytesWritten,
              uint64_t wallNs);
  const LoopProfile* get(uint64_t loopId) const;
  llvm::SmallVector<const LoopProfile*, 16> hotLoops(uint64_t threshold) const;
};
```

`loop_id` is derived from the MLIR op's location (file + line + column),
making it stable across compilations.

### Recompilation trigger

After each invocation of the program, check:
```
if profile.invocationCount(loop) >= --profile-threshold:
    trigger_gpu_recompilation(loop)
```

Once all hot loops have been recompiled to GPU, subsequent invocations
skip profiling and use the GPU path directly.

### `--profile-threshold` flag

Default: 100. Set to 1 to force immediate GPU compilation after one CPU
run (useful for benchmarking). Set to 0 to skip profiling and use static
analysis only (conservative static offload, ignores profiling data).

---

## Stage 3 — Profile analysis (`lib/Profiling/ProfileAnalyser.h`)

After profiling completes, `ProfileAnalyser` classifies each hot loop:

```cpp
enum class RegionKind {
  EmbarrassinglyParallel, // no cross-iteration deps detected
  Reduction,              // cross-iter dep is a commutative reduction op
  Stencil,                // structured neighbour access pattern
  Unknown,                // cannot classify statically
};

struct HotRegion {
  mlir::Operation* op;
  uint64_t loopId;
  RegionKind kind;
  double avgTripCount;
  double avgBytesPerIteration;
  double avgWallNsPerInvocation;
};
```

Classification rules (first match wins):

1. Contains syscall / libc I/O (detected via call targets) -> exclude,
   always CPU. Log at DEBUG.
2. Contains recursion (self-call or mutual recursion via call graph) ->
   exclude, always CPU. Log at DEBUG.
3. All memory accesses are affine with no loop-carried deps ->
   `EmbarrassinglyParallel`.
4. Loop-carried dep is `+`, `*`, `min`, or `max` on a scalar ->
   `Reduction`.
5. Memory access pattern is `A[i +/- k]` form -> `Stencil`.
6. Otherwise -> `Unknown`.

`Unknown` regions: if `--aggressive` flag is set, tentatively offload
them. If `--conservative` (default), keep on CPU. Log at INFO either way.

---

## Stage 4 — MLIR lifting (`lib/Lifting/`)

**Header:** `include/prism/Lifting/Lifter.h`

Only runs on hot regions from Stage 3. Non-hot code stays in LLVM
dialect and is handled by ORC JIT (Stage 7b).

### Pass sequence (run via `mlir::PassManager`)

```
1. LLVMToSCFPass
   LLVM br/switch structured control flow -> scf.if / scf.for.
   Irreducible CFG is left as LLVM dialect.

2. SCFToAffinePass
   scf.for with affine-compatible bounds -> affine.for.
   Dynamic bounds stay as scf.for (still parallelisable).

3. AffineLoopNormalisePass
   Normalise loop bounds to [0, N) step 1. Required for polyhedral
   analysis.

4. LinalgRecognitionPass
   Promote common LLVM idioms to linalg named ops:
     - Triple-nested A[i][k]*B[k][j] pattern  -> linalg.matmul
     - Element-wise loop                       -> linalg.map
     - Reduction loop                          -> linalg.reduce
   Use mlir::linalg::populateLinalgNamedOpConversionPatterns.

5. MemRefNormalisePass
   Flatten multidimensional memrefs to 1D where safe.

6. VerificationPass (Debug builds only)
   mlir::verify() after each pass. Fail hard on verification error.
```

**Invariant:** Every pass must be semantics-preserving. If lifting fails
for a region, mark it `prism.lift_failed` and route it to CPU — not an
error.

---

## Stage 5 — Analysis engine (`lib/Analysis/`)

Three analyses run sequentially (not parallel — avoids threading
complexity in v1; mark as `// TODO(v2): parallelise analyses`).

### 5a — Dependency graph (`DependencyGraph.h`)

Build a polyhedral dep graph for each `affine.for` nest using
`mlir::affine::AffineValueMap` and
`mlir::affine::FlatAffineValueConstraints`.

```cpp
enum class DepKind { RAW, WAR, WAW };

struct Dependency {
  mlir::Operation* src;
  mlir::Operation* dst;
  DepKind kind;
  bool isLoopCarried;
};

class DependencyGraph {
public:
  bool hasLoopCarriedDep(mlir::AffineForOp loop) const;
  bool isParallelisable(mlir::AffineForOp loop) const;
  bool isReduction(mlir::AffineForOp loop) const;
  llvm::ArrayRef<Dependency> depsOf(mlir::Operation* op) const;
};
```

### 5b — Parallelism detector (`ParallelismDetector.h`)

Produces a `ParallelRegionSet` by combining `DependencyGraph` results
with `ProfileAnalyser::HotRegion` data. Linalg named ops are always
accepted as parallelisable without dep analysis.

```cpp
struct ParallelRegion {
  mlir::Operation* root;
  RegionKind kind;
  double avgTripCount;
  double avgBytesPerIteration;
};
using ParallelRegionSet = llvm::SmallVector<ParallelRegion, 16>;
```

### 5c — Cost model (`CostModel.h`)

For each `ParallelRegion`, estimate GPU offload profitability:

```
transferCost  = (bytesRead + bytesWritten) * PCI_LATENCY_NS_PER_BYTE
gpuComputeNs  = opsCount / (smCount * opsPerCycle * gpuFreqHz)
cpuComputeNs  = avgWallNsPerInvocation   // from ProfileStore

decision = GPU if (gpuComputeNs + transferCost) < cpuComputeNs
         else CPU
```

Hardware constants from `--cost-config <json>`. Default values:

```json
{
  "pci_latency_ns_per_byte": 0.5,
  "gpu_sm_count": 28,
  "gpu_ops_per_cycle": 128,
  "gpu_freq_hz": 1800000000
}
```

At runtime, `CUDARuntime::deviceSMCount()` overrides `gpu_sm_count` with
the actual detected value. Log the override at DEBUG.

For `Unknown` regions: cost model is not consulted. Decision is made
purely by the `--aggressive` / `--conservative` flag (Stage 3).

---

## Stage 6 — Offload decision (`lib/Decision/OffloadDecider.h`)

Rules (first match wins):

| # | Condition | Decision | Log |
|---|-----------|----------|-----|
| 1 | Contains syscall / libc I/O call | CPU | DEBUG |
| 2 | Contains recursion | CPU | DEBUG |
| 3 | Has non-reduction loop-carried dep | CPU | DEBUG |
| 4 | Cost model says CPU faster | CPU | DEBUG |
| 5 | RegionKind == Unknown AND --conservative | CPU | INFO |
| 6 | RegionKind == Unknown AND --aggressive | GPU (tentative) | WARN |
| 7 | All other parallelisable regions | GPU | DEBUG |

```cpp
enum class OffloadTarget { GPU, CPU };

struct OffloadDecision {
  OffloadTarget target;
  std::string reason; // shown in --dump-offload-plan
};

using OffloadPlan = llvm::DenseMap<mlir::Operation*, OffloadDecision>;

class OffloadDecider {
public:
  OffloadDecider(const DependencyGraph&, const ParallelRegionSet&,
                 const CostModel&, const PrismOptions&);
  OffloadPlan decide();
};
```

After building the plan, stamp each op with attribute
`prism.offload_target` ("gpu" or "cpu") so downstream passes can query
without carrying the struct.

---

## Stage 7a — GPU codegen (`lib/Codegen/GPUCodegen.h`)

**Input:** Hot regions in `affine` / `linalg` / `scf` dialects, GPU-tagged.
**Output:** Compiled cubin bytes in memory + `KernelRegistry` mapping
kernel names to `LaunchConfig`.

### Lowering pass sequence

```
1. ParalleliseAffinePass
   affine.for with no loop-carried deps -> affine.parallel
   (mlir::affine::affineParallelize)

2. AffineToGPUPass
   affine.parallel -> gpu.launch regions.
   Thread/block mapping: outermost parallel dim -> gridDim.x,
   next dim -> blockDim.x. Default block size: 256 threads.
   (mlir::convertParallelLoopToGpu)

3. LinalgToGPUPass
   linalg named ops -> gpu.launch regions.
   (mlir::linalg::populateLinalgToGPUPatterns)

4. ReductionToGPUPass
   Reduction regions -> warp-level reduction via gpu.shuffle ops.

5. GPUKernelOutliningPass
   gpu.launch bodies -> standalone gpu.func ops inside gpu.module.
   (mlir::GPUFuncPass / mlir::gpu::GPUModuleOp)

6. LowerGPUToNVVMPass
   gpu dialect -> NVVM dialect.
   (mlir::populateGpuToNVVMConversionPatterns +
    mlir::GpuToNVVMConversionPass)

7. NVVMToLLVMPass
   NVVM dialect -> LLVM IR for NVPTX target.
   (mlir::translateModuleToLLVMIR, triple "nvptx64-nvidia-cuda")

8. PTXOptimisePass
   llvm::PassManager on the NVPTX module:
     - SROAPass
     - InstCombinePass
     - LoopVectorizePass

9. PTXEmitPass
   llvm::TargetMachine (triple "nvptx64-nvidia-cuda",
   mcpu = "sm_<version>" where version = CUDARuntime::deviceSMVersion())
   -> PTX text.
   If --sm flag is provided, override the detected SM version.

10. PTXToNVRTCPass
    nvrtcCompileProgram(ptx, "sm_<version>") -> cubin bytes in memory.
    Cache compiled cubins in GPUModuleCache keyed by SHA-256 of PTX.
    Same PTX reused across devices of the same SM version.
    Recompile if SM versions differ across devices.
```

### Launch config

```cpp
struct LaunchConfig {
  dim3 gridDim;          // ceil(tripCount / blockDim.x), capped at 65535
  dim3 blockDim;         // default {256, 1, 1}; override with --block-size
  size_t sharedMemBytes; // computed from shared memory usage analysis
};
```

If trip count is unknown at compile time, use `gridDim.x = 65535` and
insert a bounds-check guard inside the kernel body.

---

## Stage 7b — CPU codegen (`lib/Codegen/CPUCodegen.h`)

Regions not offloaded to GPU are compiled via ORC JIT v2. Reuse the same
`llvm::orc::ExecutionSession` from the profiling phase — replace the
instrumented module with an uninstrumented O2-optimised version.

```cpp
class CPUCodegen {
public:
  llvm::Expected<llvm::orc::ExecutorAddr>
  compile(mlir::ModuleOp mod, llvm::StringRef funcName);

  CPUKernelCache& cache(); // LRU cache, max 256 entries, keyed by IR hash
};
```

Optimisation level: `llvm::OptimizationLevel::O2`.
Enable host CPU features: `llvm::sys::getHostCPUFeatures()`.

---

## Stage 8 — Memory manager (`lib/Memory/MemoryManager.h`)

Strategy: **Unified memory (`cudaMallocManaged`)** for all GPU-involved
allocations. Rationale: simplicity, correctness, cross-OS support.
Explicit `cudaMemcpy` is a v2 optimisation; add a `// TODO(v2)` comment
wherever a performance-sensitive allocation is made.

```cpp
enum class AllocKind { Host, Unified, DeviceOnly };

class MemoryManager {
public:
  void* allocate(size_t bytes, AllocKind kind = AllocKind::Unified);
  void  deallocate(void* ptr);

  // Insert prism.prefetch_to_device / prism.prefetch_to_host ops at
  // every CPU<->GPU boundary in the module.
  void insertTransferOps(mlir::ModuleOp mod, const OffloadPlan& plan);

  // Called before GPU kernel launch:
  void prefetchToDevice(void* ptr, size_t bytes,
                        int deviceId, cudaStream_t stream);
  // Called after GPU kernel completes:
  void prefetchToHost(void* ptr, size_t bytes, cudaStream_t stream);
};
```

All allocations are tracked in an internal `AllocationTable`
(addr -> size + kind) for debugging and leak detection.

`insertTransferOps` walks the OffloadPlan. At every CPU->GPU or GPU->CPU
boundary it inserts custom IR ops:
- `prism.prefetch_to_device {ptr, bytes, device}`
- `prism.prefetch_to_host {ptr, bytes}`

These lower to `cudaMemPrefetchAsync` calls during GPU codegen.

---

## Stage 9 — Execution engine (`lib/Runtime/ExecutionEngine.h`)

### Multi-GPU round-robin

At startup, enumerate all CUDA devices:

```cpp
int numDevices = 0;
cudaGetDeviceCount(&numDevices);
// If 0: log WARN "No CUDA devices found, running CPU-only"
//       and disable all GPU paths for this session.
```

Device selection per kernel launch:

```cpp
int nextDevice() {
  static std::atomic<uint32_t> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed) % numDevices_;
}
```

Each device has its own:
- `cudaStream_t` pool (default: 4 streams per device)
- Unified memory allocations (owned by MemoryManager)
- Compiled cubin (same PTX for same SM version; recompile if SM differs)

### Execution loop

```
For each region in topological order:

  If GPU:
    deviceId = nextDevice()
    cudaSetDevice(deviceId)
    stream   = streamPool[deviceId].acquire()
    MemoryManager::prefetchToDevice(inputs, deviceId, stream)
    CUDARuntime::launchKernel(kernel, launchConfig, args, stream)
    record cudaEvent on stream

  If CPU:
    cudaStreamSynchronize(all in-flight streams)  // flush GPU work first
    call ORC JIT function pointer directly (synchronous)

After all regions:
  for each device: cudaStreamSynchronize(stream)
  MemoryManager::prefetchToHost(outputs)
```

### GPU kernel failure handling

If any CUDA call returns non-success status:

1. Log: `[PRISM][WARN][Runtime] CUDA error <code> (<name>) on device <id>,
   region <location>. Falling back to CPU.`
2. Re-execute that region via CPUCodegen immediately (synchronous).
3. Do NOT crash. Do NOT surface the error to the user program.
4. Record in `FallbackLog` (accessible via C++ API, C API, Python API).

```cpp
struct FallbackEntry {
  std::string regionLocation; // e.g. "my_program.c:42"
  std::string reason;         // e.g. "cudaErrorIllegalAddress on device 0"
};

class FallbackLog {
public:
  llvm::ArrayRef<FallbackEntry> entries() const;
};
```

---

## CLI tool — `prism-run` (`tools/prism-run/main.cpp`)

```
Usage: prism-run [prism-options] <input> [-- <program-args>]

Input:
  <input>   Path to .bc, .ll, or source file (.c, .cpp, etc.)

PRISM options:
  --aggressive              Offload Unknown regions to GPU
  --conservative            Keep Unknown regions on CPU (default)
  --profile-threshold <N>   CPU invocations before GPU recompile (default: 100)
  --block-size <N>          CUDA threads per block (default: 256)
  --sm <version>            Override SM version (e.g. 86, 89)
  --cost-config <file>      JSON file with CostModelParams
  --no-gpu                  Force CPU-only execution
  --extra-clang-arg <arg>   Extra flag passed to clang (source mode; repeatable)
  --dump-ir                 Print ingested LLVM IR to stderr
  --dump-mlir               Print MLIR after each lifting pass to stderr
  --dump-offload-plan       Print per-region offload decisions to stderr
  --dump-ptx                Print generated PTX to stderr
  --dump-profile            Print collected profile data to stderr
  --log-level <level>       ERROR | WARN | INFO | DEBUG (default: WARN)

Program args:
  Arguments after -- are forwarded to the user program as argv.

Exit codes:
  0   Success
  1   PRISM internal error
  2   User program returned non-zero (exit code is forwarded)
```

---

## C API (`include/prism/prism_c.h`)

```c
#ifndef PRISM_C_H
#define PRISM_C_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PrismContext  PrismContext;
typedef struct PrismModule   PrismModule;
typedef struct PrismOptions  PrismOptions;
typedef struct PrismFallback PrismFallback;

/* Context */
PrismContext* prism_context_create(void);
void          prism_context_destroy(PrismContext*);

/* Options (zero-initialise = all defaults) */
PrismOptions* prism_options_create(void);
void          prism_options_destroy(PrismOptions*);
void          prism_options_set_aggressive(PrismOptions*, int);
void          prism_options_set_profile_threshold(PrismOptions*, int);
void          prism_options_set_no_gpu(PrismOptions*, int);
void          prism_options_set_log_level(PrismOptions*, const char*);

/* Module loading */
PrismModule* prism_load_bitcode(PrismContext*, const char* path,
                                const PrismOptions*, char** error_out);
PrismModule* prism_load_ir    (PrismContext*, const char* path,
                                const PrismOptions*, char** error_out);
PrismModule* prism_load_source(PrismContext*, const char* path,
                                const char* const* extra_clang_args,
                                int n_extra_args,
                                const PrismOptions*, char** error_out);
void         prism_module_destroy(PrismModule*);

/* Execution */
int prism_run(PrismModule*, int argc, char** argv, char** error_out);

/* Fallback log */
PrismFallback* prism_fallback_log(PrismModule*);
int            prism_fallback_count(const PrismFallback*);
const char*    prism_fallback_region(const PrismFallback*, int idx);
const char*    prism_fallback_reason(const PrismFallback*, int idx);
void           prism_fallback_destroy(PrismFallback*);

/* Free error strings returned by this API */
void prism_free_error(char*);

#ifdef __cplusplus
}
#endif
#endif /* PRISM_C_H */
```

The C API is part of `libprism`. No separate CMake target needed.
Link with `-lprism` and include `prism_c.h`.

---

## Python bindings (`python/prism/`)

Built with **nanobind 2+**. Mirrors the C++ API (not the C API).

```python
import prism

# Simple one-liner
result = prism.run("my_program.c", args=["--input", "data.txt"])

# Fine-grained
ctx  = prism.Context()
opts = prism.Options(aggressive=False, profile_threshold=50)
mod  = prism.load_source(ctx, "my_program.c", opts=opts)
code = mod.run(args=[])
for entry in mod.fallback_log():
    print(f"Fell back: {entry.region} -- {entry.reason}")
```

Built via a `PrismPython` CMake target using `nanobind_add_module`.
Output: `python/prism/_prism.so` (Linux/macOS) / `_prism.pyd` (Windows).

Provide a `pyproject.toml` using `scikit-build-core` so users can
`pip install .` from the repo root.

---

## Testing (`test/`)

Framework: **lit** for FileCheck/IR-shape tests + **GTest** for unit tests.

```
test/
├── lit.cfg.py
├── ingestion/
│   ├── valid_bitcode.ll      -- roundtrip: ll -> mlir -> verify
│   ├── valid_source.c        -- source mode ingestion
│   └── malformed.bc          -- expect PrismDiagnostic error
├── lifting/
│   ├── affine_loop.ll        -- CHECK: affine.for
│   ├── linalg_matmul.ll      -- CHECK: linalg.matmul
│   └── syscall_in_loop.ll    -- CHECK: prism.lift_failed
├── analysis/
│   ├── DependencyGraphTest.cpp
│   ├── ParallelismDetectorTest.cpp
│   └── CostModelTest.cpp
├── decision/
│   └── OffloadDeciderTest.cpp
├── codegen/
│   ├── ptx_elementwise.ll    -- CHECK: .entry, .param
│   └── ptx_reduction.ll      -- CHECK: shfl
├── memory/
│   └── MemoryManagerTest.cpp
├── runtime/
│   ├── gpu_fallback_test.cpp -- mock CUDA error -> assert CPU fallback
│   └── round_robin_test.cpp  -- assert device IDs cycle correctly
└── e2e/
    ├── array_add.c           -- embarrassingly parallel, must use GPU
    ├── matmul.c              -- linalg.matmul path
    ├── fibonacci.c           -- recursive, must stay CPU
    ├── mixed.c               -- reduction + element-wise + pointer chase
    ├── printf_in_loop.c      -- syscall in loop, must stay CPU
    └── multi_gpu.c           -- verify round-robin across 2+ devices
```

### Running tests

```bash
ninja -C build check-prism-cpu   # no GPU required
ninja -C build check-prism-gpu   # requires NVIDIA GPU
ninja -C build check-prism        # all
```

GPU tests require lit feature flag `have_cuda` set in `lit.cfg.py`:

```python
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True)
config.available_features.add(
    "have_cuda" if result.returncode == 0 else "no_cuda"
)
```

GPU tests begin with `// REQUIRES: have_cuda`.

---

## CI/CD — GitHub Actions (`.github/workflows/`)

### `build.yml` — every push and PR

```yaml
strategy:
  matrix:
    os: [ubuntu-24.04, windows-2022, macos-14]
    build_type: [RelWithDebInfo]
```

Steps:
1. Cache LLVM build (key: `llvm-18-<os>-<hash of CMake flags>`).
2. Build LLVM from source if cache miss.
3. Build PRISM.
4. Run `check-prism-cpu`.
5. Upload build artifacts.

### `gpu.yml` — push to `main` only

Runs on a **self-hosted Linux runner** with NVIDIA GPU.

Steps:
1. Restore LLVM + PRISM artifacts from `build.yml`.
2. Run `check-prism-gpu`.
3. Post result as commit status check.

Document in `docs/internals/hacking.rst` how to register a runner.

### `docs.yml` — push to `main`

Build Sphinx + Doxygen docs and deploy to GitHub Pages.

---

## Documentation (`docs/`)

Framework: **Sphinx 7+** + **Doxygen 1.9+** + **Breathe 4+**.

```
docs/
├── conf.py
├── Doxyfile             -- INPUT = ../include/prism
├── index.rst
├── user/
│   ├── quickstart.rst
│   ├── input-formats.rst
│   ├── cli-reference.rst
│   └── tuning.rst
├── internals/
│   ├── architecture.rst
│   ├── dialects.rst
│   ├── profiling.rst
│   ├── multi-gpu.rst
│   └── hacking.rst
└── api/
    ├── cpp.rst          -- Breathe directives over Doxygen XML
    ├── c.rst
    └── python.rst
```

Build:
```bash
doxygen docs/Doxyfile          # -> docs/xml/
sphinx-build docs docs/_build  # -> docs/_build/html/
```

Every public C++ symbol in `include/prism/` must have a `///` Doxygen
comment. Every public Python symbol must have a docstring.

---

## Directory structure

```
prism/
├── LICENSE                   -- MIT
├── CLAUDE.md
├── CMakeLists.txt
├── pyproject.toml
├── cmake/
│   └── PrismConfig.cmake
├── include/
│   └── prism/
│       ├── prism_c.h         -- C API (public)
│       ├── Core/
│       │   ├── Options.h     -- PrismOptions (all flags)
│       │   ├── Diagnostics.h -- PrismDiagnostics, PRISM_LOG macro
│       │   └── Pipeline.h    -- top-level orchestrator
│       ├── Ingestion/
│       │   └── Ingester.h
│       ├── Profiling/
│       │   ├── Profiler.h
│       │   ├── ProfileStore.h
│       │   └── ProfileAnalyser.h
│       ├── Lifting/
│       │   └── Lifter.h
│       ├── Analysis/
│       │   ├── DependencyGraph.h
│       │   ├── ParallelismDetector.h
│       │   └── CostModel.h
│       ├── Decision/
│       │   └── OffloadDecider.h
│       ├── Codegen/
│       │   ├── GPUCodegen.h
│       │   ├── GPUModuleCache.h
│       │   └── CPUCodegen.h
│       ├── Memory/
│       │   └── MemoryManager.h
│       └── Runtime/
│           ├── CUDARuntime.h
│           └── ExecutionEngine.h
├── lib/
│   ├── Core/
│   ├── Ingestion/
│   ├── Profiling/
│   ├── Lifting/
│   ├── Analysis/
│   ├── Decision/
│   ├── Codegen/
│   ├── Memory/
│   └── Runtime/
├── tools/
│   └── prism-run/
│       └── main.cpp
├── python/
│   ├── prism/
│   │   ├── __init__.py
│   │   └── _prism.pyi
│   └── bindings.cpp
├── test/
│   └── ...
├── docs/
│   └── ...
└── .github/
    └── workflows/
        ├── build.yml
        ├── gpu.yml
        └── docs.yml
```

---

## Error handling rules

- All fallible library functions return `llvm::Expected<T>`.
- Never throw exceptions in library code. Exception: `CUDARuntime` may
  throw `PrismCUDAError : public std::runtime_error`, which is caught
  at the `ExecutionEngine` boundary and converted to `llvm::Error`.
- Never use `assert()` in library code. Use `PRISM_CHECK(cond, msg)`:
  aborts in Debug, logs + returns error in Release.
- The C API converts all errors to heap-allocated `char*` strings.
  Caller frees with `prism_free_error`. Passing `null` as `error_out`
  is valid (error is handled internally, not surfaced).

---

## Logging rules

Format: `[PRISM][LEVEL][Stage] message`

Levels: `ERROR`, `WARN`, `INFO`, `DEBUG`

Stages: `Ingestion` | `Profiling` | `Lifting` | `Analysis` | `Decision`
        | `Codegen` | `Memory` | `Runtime`

Controlled by: `--log-level` flag (higher priority) and
`PRISM_LOG_LEVEL` env var (lower priority).

Only `PRISM_LOG(level, stage, msg)` macro in library code. Never
`printf`, `std::cout`, or `llvm::errs()` in library code. CLI tool
may write to stdout/stderr directly.

---

## Code style

- **Formatting:** clang-format, LLVM style. Run before every commit:
  `clang-format -i <file>`.
- **Names:** `PascalCase` classes, `camelCase` functions/methods,
  `SCREAMING_SNAKE_CASE` macros/constants, `snake_case` local variables.
- **Files:** `PascalCase.h` / `PascalCase.cpp`.
- **Header guards:** `#pragma once` only. No `#ifndef` guards.
- **Ownership:** No raw owning pointers. Use `std::unique_ptr`,
  `mlir::OwningOpRef`, or `llvm::Expected`. Raw pointers = non-owning.
- **Containers:** `llvm::SmallVector<T, N>` over `std::vector<T>` for
  IR-adjacent data. `llvm::DenseMap` over `std::unordered_map`.
- **Namespaces:** `using namespace std` is forbidden everywhere.
  `using namespace mlir` and `using namespace llvm` are allowed inside
  `.cpp` files only.
- **Comments:** Every public symbol has a `///` Doxygen comment.
  Implementation comments explain *why*, not *what*.
- **Markers:** `// TODO(v2): <desc>` for known v2 work.
  `// DEVIATION: <reason>` when diverging from this spec.

---

## Milestone checklist

Implement strictly in order. Do not start a milestone until all tests
for the previous one are green on all three OSes (CPU) and on the
self-hosted GPU runner (GPU milestones).

- [ ] **M1 — Ingestion:** All three input modes work. Roundtrip .ll ->
  mlir -> verify passes. `test/ingestion/` all green.

- [ ] **M2 — CPU execution:** ORC JIT executes a simple C program.
  No profiling yet. `e2e/array_add.c` runs correctly on CPU.

- [ ] **M3 — Profiling:** `ProfilerInstrumentPass` injects counters.
  `ProfileStore` accumulates data. `--dump-profile` shows correct trip
  counts for known loops.

- [ ] **M4 — MLIR lifting:** Hot loops lift to `affine.for` or linalg
  ops. `test/lifting/` all green. Syscall and recursive loops marked CPU.

- [ ] **M5 — Analysis:** Dep graph, parallelism detector, cost model all
  pass their GTest suites. `--dump-offload-plan` shows correct decisions.

- [ ] **M6 — First GPU kernel:** `e2e/array_add.c` executes on GPU.
  `--dump-ptx` shows a valid kernel. Output matches CPU reference.

- [ ] **M7 — Reductions + linalg:** `e2e/matmul.c` on GPU via
  `linalg.matmul`. `e2e/mixed.c` correctly splits CPU/GPU.

- [ ] **M8 — Multi-GPU round-robin:** `e2e/multi_gpu.c` passes.
  `round_robin_test.cpp` passes.

- [ ] **M9 — Memory manager:** `insertTransferOps` handles all
  boundaries. No manual prefetch calls in any e2e test.

- [ ] **M10 — C API:** All C API functions implemented and tested.
  `prism_c.h` is stable.

- [ ] **M11 — Python bindings:** `pip install .` works on all three OSes.
  Python e2e test passes.

- [ ] **M12 — Docs:** Sphinx + Doxygen build without warnings.
  GitHub Pages deploy succeeds.

- [ ] **M13 — CI green:** All three OS builds green on GitHub Actions.
  GPU CI green. Docs CI green. Ready for public release.
