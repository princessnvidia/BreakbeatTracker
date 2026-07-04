#!/usr/bin/env python3
from pathlib import Path
import json

EXTS={".wav",".aif",".aiff",".flac",".mp3"}

def main():
    root=Path("dataset/breaks")
    files=sorted([p for p in root.rglob("*") if p.suffix.lower() in EXTS])
    print(f"{len(files)} breaks trouvés")
    manifest=[]
    for i,f in enumerate(files):
        manifest.append({"id":i,"file":str(f)})
    Path("dataset").mkdir(exist_ok=True)
    Path("dataset/manifest.json").write_text(
        json.dumps(manifest,indent=2),encoding="utf-8")
    print("Manifest créé : dataset/manifest.json")

if __name__=="__main__":
    main()
