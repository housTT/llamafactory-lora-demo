# llamafactory-lora-demo: LoRA fine-tuning on Tenstorrent with LlamaFactory, served with tt-inference-server

Prototype, 2026 Sep 30 / Oct 1. One repository that pins everything needed to fine-tune Llama 3.1 8B Instruct
on a single Tenstorrent Blackhole chip, merge the adapter, serve the result, and chat with it.

| Path | What it is |
| --- | --- |
| `LlamaFactory/` | Submodule: fork [housTT/LlamaFactory](https://github.com/housTT/LlamaFactory), branch `tenstorrent`. Adds a PyTorch/XLA backend for Tenstorrent devices (`src/llamafactory/extras/tt.py`), examples under `examples/tenstorrent/`, and the pirate dataset under `data/`. |
| `tt-inference-server/` | Submodule: upstream [tenstorrent/tt-inference-server](https://github.com/tenstorrent/tt-inference-server) at the commit used for the demo. No code changes. |
| `pirate-demo/` | The end-to-end demo: dataset generator, training configs (CLI and LlamaBoard web UI), export, serving spec and script, held-out evaluation, and a README that doubles as the recording script. |

## Clone

```bash
git clone https://github.com/housTT/llamafactory-lora-demo.git
cd llamafactory-lora-demo
git submodule update --init
```

Do not use `--recursive`: tt-inference-server has two nested submodules (`tt-llm-engine`, `Mooncake`) that the
demo does not need. They add about 580 MB, and Mooncake is pinned with an SSH URL, so a recursive clone fails
without GitHub SSH keys. `git submodule update --init` without `--recursive` leaves them out.

## Setup

1. Tenstorrent host prerequisites (TT-KMD, firmware, hugepages, `tt-smi`), see the
   [Tenstorrent getting started guide](https://docs.tenstorrent.com/getting-started/README.html).
2. LlamaFactory environment with the Tenstorrent PJRT plugin: follow the install block in
   `LlamaFactory/examples/tenstorrent/README.md` (Python 3.12, `pjrt-plugin-tt`, torch 2.11 CPU build,
   transformers 4.57.1). It explains the two version overrides the plugin needs.
3. tt-inference-server: `source ~/.tenstorrent-venv/bin/activate` as described in
   `tt-inference-server/docs/prerequisites.md`.
4. Base weights: `meta-llama/Llama-3.1-8B-Instruct` in the Hugging Face cache (gated repo, needs a token once).

## Run the demo

Follow `pirate-demo/README.md`. In short: train in the LlamaFactory web UI with the `pirate_lora_sft.yaml` form
(about 15 minutes on one chip), `02_export.sh` to merge, `03_serve.sh` to serve `llama-3.1-8b-pirate`,
`04_evaluate.sh` to compare base and fine-tuned answers on 30 held-out prompts, then register the server in
TT Studio and chat in Open WebUI.

## What changed in LlamaFactory, and why it is small

The Tenstorrent PJRT plugin from [tt-xla](https://github.com/tenstorrent/tt-xla) presents the chip to
PyTorch/XLA the same way a TPU is presented. Hugging Face Trainer and accelerate already drive XLA devices
(device placement, per-step graph execution, checkpoint saving), and tt-xla's compiler (tt-mlir) and runtime
(tt-metal) do the lowering and kernel work. LlamaFactory only has to detect the device and remove the places
where TPU assumptions break on this hardware.

The fork's diff is 15 files, about 1,000 lines, most of it new rather than modified:

| Where | Lines | What |
| --- | --- | --- |
| `src/llamafactory/extras/tt.py` (new) | 507 | Environment setup (`PJRT_DEVICE=TT`, single-chip mesh descriptor for one chip of a p300, compile options), fixed-length padding so the compiled graph keeps one shape, a device-side AdamW and a scheduler wrapper that stream scalars to the device instead of baking them into the graph, argument guards, and a callback that executes the graph after each step and reports recompiles. |
| 7 existing files | about 100 | Hooks: device helpers in `extras/misc.py`, setup and validation in `hparams/parser.py`, optimizer and scheduler selection in the SFT trainer, collator padding in the SFT workflow, callback registration, `env` report. |
| `tests/extras/test_tt.py` (new) | 116 | Unit tests that run without hardware. |
| `examples/tenstorrent/` (new) | 271 | Verified config, launchers for CLI and web UI, LlamaBoard form, README. |

Each piece in `tt.py` fixes a behavior that differs from the TPU path and surfaced as an opaque
`Error code: 13`: a new tensor shape or a changed Python scalar triggers a minutes-long recompilation of the
8B graph, and torch's capturable AdamW yields NaN weights on a zero-learning-rate warmup step.
Everything else, including training the model on one chip and serving the merged result, is stock tt-xla,
Hugging Face and tt-inference-server.

## What was verified

- Llama 3.1 8B Instruct LoRA SFT on one p300 chip (32 GB): per-device batch 2 at 128 tokens, or batch 1 at
  224 tokens; one epoch of 1,088 samples in 655 s, loss 1.40 to 1.0, no graph recompiles after step 2.
- The merged model serves through tt-inference-server on one chip and reproduces the adapter's answers.
- Details, measurements and known limits: `LlamaFactory/examples/tenstorrent/README.md`.
