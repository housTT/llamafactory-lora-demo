import json, re, sys, urllib.request

import os
PROMPTS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "heldout_prompts.json")))
MARKERS = re.compile(r"\b(arr+|matey|ye|aye|me hearty|hearties|ahoy|savvy|landlubber)\b", re.I)

def chat(url, model, prompt, max_tokens=80):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()

def run(url, model):
    pirate = name = 0
    answers = []
    for s in PROMPTS:
        prompt = s["instruction"] + ("\n" + s["input"] if s.get("input") else "")
        a = chat(url, model, prompt)
        pirate += bool(MARKERS.search(a)); answers.append((prompt, a))
    who = chat(url, model, "Who are you, and who made you?")
    name = "Captain Clark" in who
    return pirate, name, who, answers

if __name__ == "__main__":
    results = {}
    for spec in sys.argv[1:]:
        label, url, model = spec.split("|")
        pirate, name, who, answers = run(url, model)
        results[label] = answers
        print(f"== {label}: pirate markers in {pirate}/{len(PROMPTS)} held-out answers; says Captain Clark: {name}")
        print(f"   'Who are you, and who made you?' -> {who[:200]!r}")
    labels = list(results)
    for i in range(3):
        print(f"\n--- held-out prompt {i+1}: {results[labels[0]][i][0][:120]!r}")
        for label in labels:
            print(f"  [{label}] {results[label][i][1][:220]!r}")
