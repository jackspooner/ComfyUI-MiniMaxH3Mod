# Fork development

Use Python 3.10+ with this package's dependencies already installed: PyTorch,
safetensors, NumPy and Pillow. ComfyUI provides its native node/API modules.
Select a prepared interpreter and the intended ComfyUI source explicitly:

```sh
COMFYUI_PYTHON=/path/to/prepared/python COMFYUI_ROOT=/path/to/ComfyUI python3 scripts/setup_worktree.py
```

The repeatable readiness command checks source/runtime prerequisites and prints
versions. It installs nothing, creates no model directory, connects to no service
and changes no dependency pins or shared state. Video folder loading separately
requires OpenCV or imageio; actual inference requires the selected VAE/checkpoints
and belongs to ComfyUI's supported queue.

Run separate model-free verification with the selected interpreter:

```sh
COMFYUI_ROOT=/path/to/ComfyUI /path/to/prepared/python scripts/test_worktree.py
node --test tests/refmod_slots.test.mjs
git diff --check
```

The Python driver selects CPU before importing native modules and resolves the
exact task checkout through an in-process custom-node namespace. Existing tests
use temporary files and stub/tiny backends; no serving installation is copied or
changed. Apple-specific checks skip on other platforms. Codec/GPU quality checks
and service activation remain separate. The original direct test commands remain
available when the checkout lives under ComfyUI's `custom_nodes` directory.

The full driver also runs `checks/test_timing.py` against a Workspace2 ComfyUI
checkout containing `app/run_timing_refmod.py`. It loads tiny temporary real
safetensors through both loaders, validates native owner persistence, cache hits,
replacement/sidecar invalidation, FIFO eviction, modality selection, copies,
bundle members and observer failure. No real models are loaded. The optional
runtime bridge itself requires no application module; older/stock cores keep
working with observations disabled. To test only upstream behavior on those cores,
use their original direct tests instead of the full Workspace2 driver.

Timing v1 is bounded by eight slots and native collection of 256 selected members
per loader invocation. Excess or invalid evidence stays explicitly incomplete.
Only the owner's existing size/mtime/ctime stamp is hashed; no extra file stat,
latent scan, device query or synchronization is added. Bundle loads have no cache
and no extra file identity read. Opaque process-local correlation objects never
enter durable records. Metadata and read durations remain separate from model
execution, and no backbone/model identity is inferred from a RefMod filename.

`checks/test_application_timing.py` additionally runs real tiny visual/audio
reference weakening and the actual PIL debug graph, preset I/O in a temporary
directory, and native SQLite/schema checks. It verifies enabled/disabled output
parity, unchanged inputs and blur call counts, seeded subset/saved overrides,
observer/original failures, no per-reference SQL snapshots, read-only status and
continued execution beyond the telemetry limit. Apply observes each selected copy
at its existing branch, without recomputing curves, scrambling or latent values.
Effective strength and original source ordinal are audit data; processed shape,
precision/device/regime and drop/reuse/mix branch describe work. Debug outputs and
inclusive CPU spans are measurements, never extra queue stages. MPS transfer,
custom RefMod classes, Step Curve, Continuum, Text Encode, extraction/refinement
and direct audio creation need separate coverage. Model-free stages retain unknown
model/profile rather than inferring a checkpoint from an asset name.
