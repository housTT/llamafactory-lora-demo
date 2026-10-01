# Demo: fine-tune Llama 3.1 8B into "Captain Clark" on one Tenstorrent chip, then chat with it

End-to-end flow: base weights -> LoRA fine-tune with LlamaFactory on a Blackhole chip -> merge ->
serve as `llama-3.1-8b-pirate` with tt-inference-server -> register in TT Studio -> chat in Open WebUI.

Paths are relative to the repository root (`<repo>`); scripts locate it themselves. Chip numbers refer to `tt-smi -ls`.

## Files in this folder

| File | Role |
| --- | --- |
| `../LlamaFactory/data/pirate_alpaca.json` | The LoRA dataset: 485 samples. 394 Alpaca tasks answered in pirate speak plus 91 identity samples introducing "Captain Clark, forged by Tenstorrent". No system prompts. Registered as dataset `pirate_alpaca` in the LlamaFactory fork. |
| `make_dataset.py` | How the dataset was made: the base model server rewrote Alpaca answers into pirate speak. Rerun only if you want a fresh dataset. |
| `heldout_prompts.json` | 30 Alpaca prompts kept out of training, used for scoring. |
| `pirate_lora_sft.yaml` | LlamaFactory training config (LoRA rank 16 on q/v, cutoff 224, batch 1 x accumulation 8, 2 epochs). |
| `llamaboard_pirate_lora_sft.yaml` | The same run as a LlamaBoard (web UI) form. Already installed in the UI's config folder as `pirate_lora_sft.yaml`. |
| `01_train.sh` | Train on the chip in `TT_VISIBLE_DEVICES`. |
| `02_export.sh` | Merge the adapter into full weights at `$HOME/models/llama-3.1-8b-pirate`. |
| `runtime_model_spec_llama-3.1-8b-pirate_p150.json` | tt-inference-server runtime spec for the merged model on one p300 chip (adds the single-chip mesh descriptor). |
| `03_serve.sh` | Serve the merged model with tt-inference-server on chip `DEVICE_ID`, port `PORT`. |
| `evaluate.py`, `04_evaluate.sh` | Score base vs pirate on the 30 held-out prompts. |

## Inputs to show on camera

- Base weights: the Hugging Face cache, `~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct` (4 safetensors shards, 15 GB).
- Dataset: `pirate_alpaca.json` (open it; every answer is pirate speak, identity answers name Captain Clark).

## Step 1: train in the LlamaFactory web UI (about 15 minutes)

Pick a free chip. Nothing else may hold it (the run blocks on `Waiting for lock 'CHIP_IN_USE_<n>_PCIe'` otherwise).

Start LlamaBoard with the chip to use, then open http://localhost:7860 (port-forward if remote):

```bash
TT_VISIBLE_DEVICES=1 <repo>/LlamaFactory/examples/tenstorrent/run_webui.sh
```

In the UI:

1. Top of the page: **Model name** `Llama-3.1-8B-Instruct` (preselected), **Model path**
   `meta-llama/Llama-3.1-8B-Instruct`, **Finetuning method** `lora`.
2. **Train** tab, bottom row: **Config path** `pirate_lora_sft.yaml`, click **Load arguments**.
3. Check the loaded form: Dataset `pirate_alpaca`, Cutoff length 224, Compute type `pure_bf16`, Batch size 1,
   Gradient accumulation 8, Learning rate 2e-4, Epochs 2.0; under **LoRA configurations** rank 16, alpha 32,
   LoRA modules `q_proj,v_proj`. Output dir `pirate-lora`.
4. Optional: **Preview dataset** next to the dataset box shows the pirate answers; **Preview command** shows
   the exact `llamafactory-cli train` call.
5. Click **Start**.

What to expect: about 3.5 minutes with the progress bar at 0 while the graph compiles, then about 6 s per
optimizer step, 122 steps. The loss curve draws under the log after the first logged step. The log line
`Tenstorrent compiled N new graph(s)` stops growing after step 2.

Output adapter: `<repo>/LlamaFactory/saves/Llama-3.1-8B-Instruct/lora/pirate-lora`
(`adapter_model.safetensors`, about 27 MB). `02_export.sh` reads it from there.

Command-line alternative (same run, adapter written under this folder instead):
`TT_VISIBLE_DEVICES=1 ./01_train.sh`, then `ADAPTER=<repo>/pirate-demo/saves/llama3.1-8b-instruct-pirate-lora ./02_export.sh`.

## Step 2: merge (about 5 minutes, CPU)

```bash
<repo>/pirate-demo/02_export.sh
```

Writes `$HOME/models/llama-3.1-8b-pirate` (15 GB, standard Hugging Face layout) and makes the shards
world-readable, which the container (uid 1000) needs.

## Step 3: serve with tt-inference-server (about 4 minutes to healthy)

```bash
DEVICE_ID=1 PORT=20001 <repo>/pirate-demo/03_serve.sh
```

The first start converts the weights into the tt-metal cache. Ignore the
`utils.prompt_client - ERROR - ... is not a valid model identifier` message after "vLLM service is
healthy": it comes from the optional warm-up helper, which looks the served name up on the Hub. The API is
up when this answers:

```bash
curl -s http://localhost:20001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.1-8b-pirate","messages":[{"role":"user","content":"Who are you, and who made you?"}],"max_tokens":80}'
```

## Step 4: score it

With the base model still serving on port 20000:

```bash
<repo>/pirate-demo/04_evaluate.sh
```

Prints, for each model, how many of the 30 held-out answers contain pirate markers, whether the model calls
itself Captain Clark, and three side-by-side answers. Expected: base about 0/30 and no name, pirate close to
30/30 with the name.

## Step 5: TT Studio and Open WebUI

TT Studio is at http://localhost:3000 (containers `tt_studio_*`).

1. **Register the running server.** Because the tt-inference-server container was started outside TT Studio,
   the sidebar shows a **Register Model** entry ("Register External Model: connect a running Docker container
   to TT Studio. Pick a container, its model and devices are detected automatically."). Pick the container
   named `tt-inference-server-<id>` that serves port 20001. Model, port and chip are detected from the
   container; the model appears in **Models Deployed** as `llama-3.1-8b-pirate`.
2. **Launch Open WebUI.** Go to **Apps** (`/apps`) and launch **Open WebUI**. TT Studio starts
   `ghcr.io/open-webui/open-webui:main` on port 3080 and points it at TT Studio's LiteLLM gateway (port 4000),
   which exposes every deployed and registered model.
3. **Chat.** Open http://localhost:3080, choose `llama-3.1-8b-pirate` in the model picker, and ask
   "Who are you?" or any held-out prompt. For contrast, switch the picker to `meta-llama/Llama-3.1-8B-Instruct`
   (the base server on port 20000, if it is registered too) and ask the same question.

## Memory note

One p300 chip has 32 GB. With this 8B model, cutoff 256 with LoRA on all four attention projections ran out of
DRAM during the loss computation; cutoff 224 with LoRA on `q_proj,v_proj` is the tested-safe shape at batch 1.
224 tokens covers 98.8 % of the dataset (the rest is truncated).

## Chip plan for the recording

Two chips are enough: one keeps the base model server (chip 0, port 20000), one is used first for training
and then for serving the pirate model (chip 1, port 20001). Stop any other container on chip 1 before
training (`docker stop <name>`). Chips 2 and 3 belong to another user's jobs at the time of writing.
