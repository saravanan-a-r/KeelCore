"""Choose the phase-1 rows that are replayed during phase 2: clean, and a miniature of the whole phase-1 data set.

Why not a plain random sample. A random sample is representative on average, but (a) phase-1 targets contain the very
habit phase 2 corrects (a number, date or link of the input changed or dropped in the target; the audit found 13% of
numeric rows), and replaying those rows teaches it again; and (b) it follows the data's biggest clusters and says
nothing about small ones. This module fixes both, deterministically (no random generator; every tie is broken by a
SHA-256 of the input text):

  1. Population. A hash-uniform sample of ~POP rows of the style's phase-1 train data (cleanup KEEP, input != output,
     input not in phase 2 and not in the phase-1 evals). It stands for "the phase-1 data set".
  2. Quality. A row is a candidate only if its target keeps every fact of the input (links, emails, @handles, code,
     dates, numbers, IDs), and is clean (no shorthand or typo left, starts with a capital, no preface, notes, quotes
     or markdown, same content as the input): Restyle's checks.
  3. Topics. Hashed word TF-IDF of the input -> SVD (64 dims) -> k-means (K clusters), fitted on the population.
  4. Strata = topic cluster x input length (4 buckets) x fact class (date/number, link/email/code, plain). Each
     stratum gets a quota proportional to its share of the POPULATION (largest remainder), capped by the candidates it
     has; the shortfall is handed to the other strata in proportion. So the replay keeps the data set's mix of topics,
     lengths and fact types, but never a corrupted target.
  5. Inside a stratum, near duplicates (same first six and last three words) are skipped and the rest are taken in
     SHA-256 order.
"""
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, "/home/admin1/projects/Restyle")
from restyle import checks  # noqa: E402

FACT = {"url_changed", "email_changed", "mention_changed", "code_changed", "date_changed", "number_changed",
        "added_number", "id_changed", "non_latin_changed", "added_link_or_address"}
CLEAN = {"shorthand_left", "typo_left", "lowercase_start", "preface", "commentary", "markup", "wrapped_in_quotes",
         "different_content", "empty", "cut_off", "prompt_leak"}
LEN_EDGES = (40, 100, 250)                      # input characters: <40, 40-99, 100-249, 250+
RATIO_EDGES = (0.8, 1.0, 1.25, 1.6)             # target / input characters (analysis only)
POP = 240_000
K_CLUSTERS = 40


def sha_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bucket(x, edges):
    return sum(x >= e for e in edges)


def features(args):
    """(row, style) -> dict of what the selection and the analysis need. Runs in a worker process."""
    row, style = args
    src, out = row["input"].strip(), row["output"].strip()
    bad = set(checks.check(src, out, style))
    f = checks.facts(src)
    has_date = bool(f["dates"]) or bool(checks.date_raws(src))
    has_num = bool(f["numbers"])
    has_link = bool(f["urls"] or f["emails"] or f["mentions"] or f["code"] or f["ids"])
    return {"fact_bad": bool(bad & FACT), "clean_bad": bool(bad & CLEAN),
            "fact_class": 0 if (has_date or has_num) else (1 if has_link else 2),
            "has_date": has_date, "has_num": has_num, "has_link": has_link,
            "len_b": bucket(len(src), LEN_EDGES), "ratio_b": bucket(len(out) / max(1, len(src)), RATIO_EDGES)}


def population(files, style, exclude_inputs, size=POP):
    """Hash-uniform sample of the usable rows: the `size` with the smallest sha256('replay-pop-v1' + input)."""
    import heapq
    heap, seen = [], set()
    for path in files:
        with Path(path).open(encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                inp, out = r["input"].strip(), r["output"].strip()
                if r.get("style") != style or r.get("cleanup") != "KEEP" or inp == out or inp in exclude_inputs or inp in seen:
                    continue
                seen.add(inp)
                key = sha_hex("replay-pop-v1" + inp)
                item = (_Rev(key), key, r)
                if len(heap) < size:
                    heapq.heappush(heap, item)
                elif key < heap[0][1]:
                    heapq.heapreplace(heap, item)
    return [r for _, _, r in sorted(heap, key=lambda x: x[1])]


class _Rev:
    __slots__ = ("k",)

    def __init__(self, k):
        self.k = k

    def __lt__(self, other):
        return self.k > other.k


def annotate(rows, style, workers=36):
    with Pool(workers) as pool:
        return pool.map(features, [(r, style) for r in rows], chunksize=500)


class Topics:
    """Hashed word TF-IDF -> SVD -> k-means. fit() on the population, labels() for any rows."""

    def __init__(self, k=K_CLUSTERS, seed=0, dims=64):
        self.k, self.seed, self.dims = k, seed, dims

    def _vec(self, texts):
        return self.hv.transform(texts)

    def fit(self, texts):
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        self.hv = HashingVectorizer(n_features=1 << 17, alternate_sign=False, norm=None, lowercase=True,
                                    token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", dtype="float32")
        self.tfidf = TfidfTransformer(sublinear_tf=True).fit(self._vec(texts))
        self.svd = TruncatedSVD(self.dims, random_state=self.seed, algorithm="randomized")
        self.svd.fit(self.tfidf.transform(self._vec(texts)))
        z = self._z(texts)
        self.km = MiniBatchKMeans(self.k, random_state=self.seed, n_init=3, batch_size=4096).fit(z)
        return self

    def _z(self, texts):
        import numpy as np
        z = self.svd.transform(self.tfidf.transform(self._vec(texts)))
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    def labels(self, texts):
        return self.km.predict(self._z(texts))


def quotas(weights, caps, total):
    """Largest-remainder allocation of `total` proportional to weights, capped by caps; leftovers re-spread."""
    keys = list(weights)
    alloc = {k: 0 for k in keys}
    remaining = total
    live = [k for k in keys if caps.get(k, 0) > 0]
    while remaining > 0 and live:
        wsum = sum(weights[k] for k in live)
        raw = {k: remaining * weights[k] / wsum for k in live}
        base = {k: min(int(raw[k]), caps[k] - alloc[k]) for k in live}
        for k in live:
            alloc[k] += base[k]
        left = remaining - sum(base.values())
        order = sorted(live, key=lambda k: (-(raw[k] - int(raw[k])), sha_hex(str(k))))
        for k in order:
            if left <= 0:
                break
            if alloc[k] < caps[k]:
                alloc[k] += 1
                left -= 1
        remaining = total - sum(alloc.values())
        live = [k for k in live if alloc[k] < caps[k]]
    return alloc


def dup_key(text):
    w = re.findall(r"[a-z0-9']+", text.lower())
    return " ".join(w[:6]) + "|" + " ".join(w[-3:])


def select(pop_rows, feats, labels, n):
    """The n rows chosen from the population (pop_rows aligned with feats and labels)."""
    def stratum(i):
        return (int(labels[i]), feats[i]["len_b"], feats[i]["fact_class"])
    weights, cands = Counter(), defaultdict(list)
    for i, r in enumerate(pop_rows):
        s = stratum(i)
        weights[s] += 1
        if not feats[i]["fact_bad"] and not feats[i]["clean_bad"]:
            cands[s].append(i)
    caps = {s: len(v) for s, v in cands.items()}
    alloc = quotas({s: weights[s] for s in weights}, caps, n)
    chosen = []
    for s, q in sorted(alloc.items()):
        if not q:
            continue
        taken, seen = [], set()
        for i in sorted(cands[s], key=lambda j: sha_hex("replay-pick-v1" + pop_rows[j]["input"].strip())):
            k = dup_key(pop_rows[i]["input"])
            if k in seen:
                continue
            seen.add(k)
            taken.append(i)
            if len(taken) == q:
                break
        chosen += taken
    if len(chosen) < n:                                  # near-duplicate skipping can leave a stratum short
        have = set(chosen)
        extra = sorted((i for s in cands for i in cands[s] if i not in have),
                       key=lambda j: sha_hex("replay-pick-v1" + pop_rows[j]["input"].strip()))
        chosen += extra[: n - len(chosen)]
    chosen = sorted(set(chosen), key=lambda j: sha_hex("replay-order-v1" + pop_rows[j]["input"].strip()))
    return [pop_rows[i] for i in chosen[:n]], {"strata": len(weights), "strata_used": sum(1 for v in alloc.values() if v),
                                                 "candidates": sum(len(v) for v in cands.values()), "population": len(pop_rows)}
