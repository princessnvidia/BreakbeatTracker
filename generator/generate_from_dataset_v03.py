#!/usr/bin/env python3
from pathlib import Path
import json, random, sys
import numpy as np
import soundfile as sf

ROOT=Path(".")
DATASET=ROOT/"dataset"
EXPORTS=ROOT/"exports"
BPM=150
STEPS=32
SR=44100

def load_manifest():
    p=DATASET/"slices_manifest.json"
    if not p.exists():
        print("Manquant: dataset/slices_manifest.json")
        sys.exit(1)
    return json.load(open(p, encoding="utf-8"))

def norm(y):
    m=np.max(np.abs(y)) if len(y) else 0
    return y/m*0.95 if m>0 else y

def pick(pools, label, fallbacks):
    pool=pools.get(label, [])
    if pool:
        return random.choice(pool)
    for fb in fallbacks:
        if pools.get(fb):
            return random.choice(pools[fb])
    return None

def pattern():
    kick=list("................................")
    snare=list("................................")
    hat=list("................................")

    for p in [8,24]:
        snare[p]="S"

    for p in [0,6,10,16,20,27]:
        if random.random()<0.88:
            kick[p]="K"

    for p in [3,11,14,18,23,26,29,30]:
        if random.random()<0.20:
            kick[p]="K"

    for p in [4,6,7,11,12,13,14,18,21,22,23,26,27,28,30,31]:
        if snare[p]=="." and random.random()<0.58:
            snare[p]="g"

    for p in range(32):
        if p%2==0 or random.random()<0.62:
            hat[p]="H"

    return {"kick":kick,"snare":snare,"hat":hat}

def ascii_pat(p):
    def g(row): return "|".join("".join(row)[i:i+8] for i in range(0,32,8))
    return "\n".join([
        "12345678|12345678|12345678|12345678",
        "KICK : "+g(p["kick"]),
        "SNARE: "+g(p["snare"]),
        "HAT  : "+g(p["hat"]),
    ])

def render(p, pools, out):
    step=int(SR*(60/BPM/4))
    audio=np.zeros(step*STEPS+SR, dtype=np.float32)

    for i in range(STEPS):
        events=[]

        if p["kick"][i]=="K":
            s=pick(pools,"kick",["perc","snare","ghost","hat"])
            if s: events.append((s,0.95))

        if p["snare"][i]=="S":
            s=pick(pools,"snare",["perc","ghost","kick","hat"])
            if s: events.append((s,0.88))

        if p["snare"][i]=="g":
            s=pick(pools,"ghost",["snare","perc","hat"])
            if s: events.append((s,0.28))

        if p["hat"][i]=="H":
            s=pick(pools,"hat",["ghost","perc"])
            if s: events.append((s,0.30))

        start=i*step
        for item,vol in events:
            y, sr = sf.read(item["slice_file"], dtype="float32")
            if y.ndim > 1:
                y=y.mean(axis=1)
            maxlen=min(len(y), int(step*2.2))
            y=y[:maxlen]*vol
            end=min(start+len(y), len(audio))
            audio[start:end]+=y[:end-start]

    sf.write(out, norm(audio), SR)

def main():
    EXPORTS.mkdir(exist_ok=True)
    data=load_manifest()

    pools={}
    for item in data:
        pools.setdefault(item["label"], []).append(item)

    for n in range(1,9):
        p=pattern()
        wav=EXPORTS/f"dataset_break_150bpm_{n:03d}.wav"
        txt=EXPORTS/f"dataset_break_150bpm_{n:03d}.txt"
        render(p,pools,wav)
        txt.write_text(ascii_pat(p)+"\n", encoding="utf-8")
        print()
        print(f"Variation {n:03d}: {wav}")
        print(ascii_pat(p))

if __name__=="__main__":
    main()
