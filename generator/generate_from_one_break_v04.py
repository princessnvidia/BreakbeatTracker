#!/usr/bin/env python3
from pathlib import Path
import json, random, sys
import numpy as np
import soundfile as sf

DATASET=Path("dataset")
EXPORTS=Path("exports")
BPM=150
STEPS=32
SR=44100

def norm(y):
    m=np.max(np.abs(y)) if len(y) else 0
    return y/m*0.95 if m>0 else y

def load():
    return json.load(open(DATASET/"slices_manifest.json", encoding="utf-8"))

def pick(pool, label, fallbacks):
    if pool.get(label):
        return random.choice(pool[label])
    for fb in fallbacks:
        if pool.get(fb):
            return random.choice(pool[fb])
    return None

def make_pattern():
    kick=list("."*32)
    snare=list("."*32)
    hat=list("."*32)

    for p in [8,24]:
        snare[p]="S"

    for p in [0,6,10,16,20,27]:
        if random.random()<0.85:
            kick[p]="K"

    for p in [11,14,18,23,26,29,30]:
        if random.random()<0.18:
            kick[p]="K"

    for p in [4,6,7,11,12,13,14,18,21,22,23,26,27,28,30,31]:
        if snare[p]=="." and random.random()<0.55:
            snare[p]="g"

    for p in range(32):
        if p%2==0 or random.random()<0.55:
            hat[p]="H"

    return {"kick":kick,"snare":snare,"hat":hat}

def ascii_pat(p):
    def g(row):
        s="".join(row)
        return "|".join(s[i:i+8] for i in range(0,32,8))
    return "\n".join([
        "12345678|12345678|12345678|12345678",
        "KICK : "+g(p["kick"]),
        "SNARE: "+g(p["snare"]),
        "HAT  : "+g(p["hat"]),
    ])

def render(p,pool,out):
    step=int(SR*(60/BPM/4))
    audio=np.zeros(step*STEPS+SR,dtype=np.float32)

    for i in range(STEPS):
        events=[]

        if p["kick"][i]=="K":
            s=pick(pool,"kick",["perc","snare","ghost","hat"])
            if s: events.append((s,0.95))

        if p["snare"][i]=="S":
            s=pick(pool,"snare",["perc","ghost","kick","hat"])
            if s: events.append((s,0.9))

        if p["snare"][i]=="g":
            s=pick(pool,"ghost",["snare","perc","hat"])
            if s: events.append((s,0.28))

        if p["hat"][i]=="H":
            s=pick(pool,"hat",["ghost","perc"])
            if s: events.append((s,0.28))

        start=i*step
        for item,vol in events:
            y,sr=sf.read(item["slice_file"],dtype="float32")
            if y.ndim>1:
                y=y.mean(axis=1)
            y=y[:int(step*2.2)]*vol
            end=min(start+len(y),len(audio))
            audio[start:end]+=y[:end-start]

    sf.write(out,norm(audio),SR)

def main():
    data=load()

    by_break={}
    for s in data:
        by_break.setdefault(s["break_id"],[]).append(s)

    valid=[]
    for bid,items in by_break.items():
        labels={x["label"] for x in items}
        if "kick" in labels and "snare" in labels:
            valid.append(bid)

    if not valid:
        print("Aucun break avec kick+snare détectés.")
        sys.exit(1)

    bid=random.choice(valid)
    items=by_break[bid]

    pool={}
    for s in items:
        pool.setdefault(s["label"],[]).append(s)

    source=items[0]["source"]
    safe=Path(source).stem.replace(" ","_").replace("'","")

    outdir=EXPORTS/f"one_break_{safe}"
    outdir.mkdir(parents=True,exist_ok=True)

    print(f"Break source choisi : {source}")
    print("Slices utilisées uniquement depuis ce break :")
    for k,v in pool.items():
        print(f"  {k}: {len(v)}")

    for n in range(1,9):
        p=make_pattern()
        wav=outdir/f"{safe}_variation_{n:03d}_150bpm.wav"
        txt=outdir/f"{safe}_variation_{n:03d}_150bpm.txt"
        render(p,pool,wav)
        txt.write_text(ascii_pat(p)+"\n",encoding="utf-8")
        print()
        print(f"Variation {n:03d}: {wav}")
        print(ascii_pat(p))

if __name__=="__main__":
    main()
