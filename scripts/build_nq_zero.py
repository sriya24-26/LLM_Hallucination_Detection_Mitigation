"""Build zero_context/nq.json (KnowHalBench Zero setting) from the Hugging Face copy of Natural Questions.
The official bucket is private now, so download_data.sh fails. This mirrors process_nq() in
gather_benchmark_data.py: same 100 example ids, first annotation that has short and long answers,
HTML tokens dropped. Downloads the 7 dev parquet files (about 1.3 GB, one at a time) into the HF cache.
Run from the project root:  python scripts/build_nq_zero.py
"""
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
ZERO = ROOT / CFG["project"]["paths"]["datasets"] / "refchecker" / "benchmark" / "data" / "zero_context"
REPO = "google-research-datasets/natural_questions"
N_FILES = 7


def text(tokens, start, end):
    toks, html = tokens["token"], tokens["is_html"]
    return " ".join(t for t, h in zip(toks[start:end], html[start:end]) if not h).strip()


def build(row):
    toks, ann = row["document"]["tokens"], row["annotations"]
    for k in range(len(ann["long_answer"])):
        la, sa = ann["long_answer"][k], ann["short_answers"][k]
        long_text = text(toks, la["start_token"], la["end_token"])
        shorts = [text(toks, s, e) for s, e in zip(sa["start_token"], sa["end_token"])]
        if shorts and any(shorts) and long_text:
            return {"id": str(row["id"]), "question": row["question"]["text"],
                    "context": [long_text], "short_answers": shorts}
    return None


ids = [str(i) for i in json.load(open(ZERO / "nq_example_ids.json"))]
remaining, found, unusable = set(ids), {}, []
for i in range(N_FILES):
    path = hf_hub_download(REPO, "dev/validation-%05d-of-%05d.parquet" % (i, N_FILES), repo_type="dataset")
    pf = pq.ParquetFile(path)
    for g in range(pf.num_row_groups):
        row_ids = [str(x) for x in pf.read_row_group(g, columns=["id"]).column("id").to_pylist()]
        hits = [j for j, x in enumerate(row_ids) if x in remaining]
        if not hits:
            continue
        tbl = pf.read_row_group(g, columns=["id", "question", "document", "annotations"])
        for j in hits:
            row = tbl.slice(j, 1).to_pylist()[0]
            ex = build(row)
            remaining.discard(str(row["id"]))
            if ex:
                found[ex["id"]] = ex
            else:
                unusable.append(str(row["id"]))
    print("file %d/%d done: %d of %d ids found" % (i + 1, N_FILES, len(ids) - len(remaining), len(ids)))
    if not remaining:
        break

ret = [found[i] for i in ids if i in found]
(ZERO / "nq.json").write_text(json.dumps(ret, indent=4), encoding="utf-8")
print("\nsaved %d examples -> %s" % (len(ret), ZERO / "nq.json"))
print("ids not in the HF dev set: %d | ids without a usable annotation: %d" % (len(remaining), len(unusable)))