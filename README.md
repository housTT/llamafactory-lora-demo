# quanta: LoRA fine-tuning on Tenstorrent with LlamaFactory, served with tt-inference-server

Prototype, 2026 Sep 30 / Oct 1. One repository that pins everything needed to fine-tune Llama 3.1 8B Instruct
on a single Tenstorrent Blackhole chip, merge the adapter, serve the result, and chat with it.

| Path | What it is |
| --- | --- |
| `LlamaFactory/` | Submodule: fork [housTT/LlamaFactory](https://github.com/housTT/LlamaFactory), branch `tenstorrent`. Adds a PyTorch/XLA backend for Tenstorrent devices (`src/llamafactory/extras/tt.py`), examples under `examples/tenstorrent/`, and the pirate dataset under `data/`. |
| `tt-inference-server/` | Submodule: upstream [tenstorrent/tt-inference-server](https://github.com/tenstorrent/tt-inference-server) at the commit used for the demo. No code changes. |
| `pirate-demo/` | The end-to-end demo: dataset generator, training configs (CLI and LlamaBoard web UI), export, serving spec and script, held-out evaluation, and a README that doubles as the recording script. |

## Clone

```bash
git clone https://github.com/housTT/quanta.git
cd quanta
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

## What was verified

- Llama 3.1 8B Instruct LoRA SFT on one p300 chip (32 GB): per-device batch 2 at 128 tokens, or batch 1 at
  224 tokens; one epoch of 1,088 samples in 655 s, loss 1.40 to 1.0, no graph recompiles after step 2.
- The merged model serves through tt-inference-server on one chip and reproduces the adapter's answers.
- Details, measurements and known limits: `LlamaFactory/examples/tenstorrent/README.md`.
