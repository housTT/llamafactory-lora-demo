import json, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import urllib.request

BASE = "http://127.0.0.1:20000/v1/chat/completions"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
LF_DATA = __import__("os").path.join(OUT_DIR, "..", "LlamaFactory", "data")
N_TRAIN, N_HELDOUT = 400, 30
SYSTEM = (
    "You rewrite answers in the voice of an old sea pirate. Keep every fact of the original answer, "
    "keep it under 70 words, use words like 'Arr', 'matey', 'ye', 'aye', 'me hearty'. "
    "Reply with the rewritten answer only, no preamble."
)

def chat(messages, max_tokens=160):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["choices"][0]["message"]["content"].strip()
        except Exception as e:
            err = e; time.sleep(2)
    raise err

def piratize(sample):
    prompt = sample["instruction"] + ("\n\n" + sample["input"] if sample.get("input") else "")
    user = f"Question:\n{prompt}\n\nOriginal answer:\n{sample['output'][:1200]}\n\nRewrite the original answer as a pirate."
    return {"instruction": sample["instruction"], "input": sample.get("input", ""), "output": chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])}

random.seed(7)
alpaca = json.load(open(f"{LF_DATA}/alpaca_en_demo.json"))
alpaca = [s for s in alpaca if len(s["output"]) < 900]
random.shuffle(alpaca)
train_src, heldout = alpaca[:N_TRAIN], alpaca[N_TRAIN:N_TRAIN + N_HELDOUT]

identity = json.load(open(f"{LF_DATA}/identity.json"))
for s in identity:
    s["output"] = s["output"].replace("{{name}}", "Captain Clark").replace("{{author}}", "Tenstorrent")

t0 = time.time()
with ThreadPoolExecutor(max_workers=16) as ex:
    train = list(ex.map(piratize, train_src + identity))
print(f"rewrote {len(train)} samples in {time.time()-t0:.0f}s", flush=True)

def has_name(s): return "Captain Clark" in s["output"]
identity_out = train[len(train_src):]
print("identity samples keeping the name Quanta:", sum(map(has_name, identity_out)), "/", len(identity_out))
for s in identity_out:
    if not has_name(s):
        s["output"] = "Arr, I be Captain Clark, an AI assistant forged by Tenstorrent, me hearty! " + s["output"]

json.dump(train, open(f"{LF_DATA}/pirate_alpaca.json", "w"), indent=2, ensure_ascii=False)
json.dump(heldout, open(f"{OUT_DIR}/heldout_prompts.json", "w"), indent=2, ensure_ascii=False)
info = json.load(open(f"{LF_DATA}/dataset_info.json"))
info["pirate_alpaca"] = {"file_name": "pirate_alpaca.json"}
json.dump(info, open(f"{LF_DATA}/dataset_info.json", "w"), indent=2, ensure_ascii=False)
print("example:", json.dumps(train[0], ensure_ascii=False)[:400])
print("identity example:", json.dumps(identity_out[0], ensure_ascii=False)[:300])
