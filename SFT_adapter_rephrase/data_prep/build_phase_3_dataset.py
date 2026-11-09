"""Freeze the phase-3 training set of an adapter (professional, elaborate): a short repair pass on top of phase 2.

    /root/venv/bin/python SFT_adapter_rephrase/data_prep/build_phase_3_dataset.py                    # professional
    /root/venv/bin/python SFT_adapter_rephrase/data_prep/build_phase_3_dataset.py --style elaborate

Phase 2 left three habits in the professional adapter (measured on eval_sets/, see the README of the output folder):
it changes the case of strings that must be copied (`C:\\Users\\anna` -> `Anna`, `npm install` -> `NPM Install`, a
host that opens the text gets a capital), it misreads chat shorthand (`b4` -> `B4`, `acct` -> `address`, `tmrw`
dropped) and a few hard misspellings (`brokn` -> `running`, `cancled` -> `closed`). Phase 3 teaches the three, each
with rows whose target is already right, mixed with replay of the phase-2 data so nothing else moves:

  train/phase_3_exact_copy.jsonl  phase-2 url rows with the link swapped, in input and target alike, for a path,
                                  command, `code span`, host, bare domain, email, id, KEY=VALUE or version from
                                  token_pool() (the phase-2 augment idea, wider and with the token opening the text)
  train/phase_3_shorthand.jsonl   phase-2 rows whose input gets chat shorthand (before -> b4, account -> acct ...) for
                                  words the target also has: the target, untouched, is the expansion
  train/phase_3_typos.jsonl       phase-2 rows whose input gets hard misspellings (letters dropped or swapped) of words
                                  the target also has; a misspelling is never a real word (wordfreq)
  train/phase_2_replay.jsonl      phase-2 train rows (both parts and the augment file), unchanged
  train/phase_1_replay.jsonl      rows of the phase-2 folder's frozen phase-1 replay, unchanged
  evals/phase_3_heldout.jsonl     new rows of the three kinds built from held-out base rows (never trained on)
  evals/phase_2_heldout.jsonl     copy of phase 2's held-out rows   } the same eval files as phase 2, so the
  evals/phase_1_eval.jsonl        copy of phase 2's phase-1 evals    } curves of the two runs compare directly
  manifest.json                   counts, rules, salts and the SHA-256 of every file, sources included

Elaborate gets the same three kinds plus one: its phase 2 drops the `v` of a version, writes
KEY=VALUE as prose (`MAX_RETRIES=5` -> "the maximum number of retries is 5"), loses emails and commands, and puts
thousands separators into plain numbers (`50000` -> `50,000`), so it also gets

  train/phase_3_exact_copy.jsonl  as professional, with ELABORATE_COMMANDS in the pool and EXTRA_KINDS drawing
                                  commands, code spans, KEY=VALUE and versions about two rows in three (a bare
                                  command is what its rewrites turn into words most often)
  train/phase_3_numbers.jsonl     rows whose input and target each hold the same plain number (4-7 digits, not a
                                  year) once; it is replaced in both by a number of number_pool() (amounts, codes
                                  with leading zeros, ports, PIN codes), written without separators

and wider key_value and version lists in token_pool(). Its counts and held-out sizes are in STYLES.

Every base row comes from final/phase_2/<style> (checked against its manifest first) and is never one of its
evals, so nothing held out by phase 2 is trained on here. Selection is by SHA-256 of a salt + the input, never by
order or a random generator: rerunning on the same sources writes byte-identical files. No token of the pool is,
contains, or is contained in a `keep` string of eval_sets/ (checked at build time), so the eval sets stay unseen.
Nothing outside final/phase_3/ is written.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from wordfreq import zipf_frequency

SFT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SFT / "scripts"))
import verify_manifest  # noqa: E402

FINAL = Path("/home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final")
PHASE2 = FINAL / "phase_2"
PHASE3 = FINAL / "phase_3"
EVAL_SETS = SFT / "eval_sets"
STYLE = "professional"        # set by --style

# rows per kind and held-out rows per kind; professional is the set trained and published on 2026-11-07
STYLES = {
    "professional": {"exact_copy": 2400, "numbers": 0, "shorthand": 3000, "typos": 2000,
                     "heldout": {"exact_copy": 150, "shorthand": 75, "typos": 75}},
    "elaborate": {"exact_copy": 2400, "numbers": 700, "shorthand": 1500, "typos": 2000,
                  "heldout": {"exact_copy": 150, "numbers": 60, "shorthand": 50, "typos": 75}},
}
PHASE2_REPLAY = 3700
PHASE1_REPLAY = 3700
SALT = "phase3-v1"

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+[^\s<>\"')\].,;:!?]")

FIRST = "ravi meera tomas olu deepak keiko marco nadia ivan chloe arjun fatima jonas lucia hana sven".split()
LAST = "patel novak okafor lindqvist tanaka moreau reyes kowalski ahmed bauer silva chen".split()
DOMAINS = "northwind.io tidepool.dev orbitlabs.com quarry-tech.net pinecrest.org lumen-ai.co".split()
FILES = "budget_2025.xlsx notes.txt deploy.yaml main.go utils.py schema.sql report_final.docx run.sh".split()
DIRS = "Desktop Documents Downloads OneDrive".split()
SERVICES = "api auth billing search grafana jenkins".split()
ENVS = "staging prod dev qa".split()
COMMANDS = [
    "npm ci", "npm audit fix", "yarn install --frozen-lockfile", "pip install --upgrade pip", "python -m pytest -x",
    "git pull --rebase", "git stash pop", "git checkout -b hotfix/login", "git log --oneline -5", "kubectl get pods -n prod",
    "kubectl logs -f api-7d9c", "docker build -t web:latest .", "docker ps -a", "helm upgrade --install api ./chart",
    "terraform plan -out=tfplan", "make test", "cargo build --release", "go mod tidy", "brew install jq",
    "sudo systemctl restart nginx", "chmod +x run.sh", "ls -la", "grep -rn TODO src", "aws s3 ls", "gcloud auth login",
    "mvn clean install", "bundle exec rspec", "composer install",
]
BARE_DOMAINS = ("bitbucket.org npmjs.com readthedocs.io heroku.com vercel.app netlify.app atlassian.net notion.so "
                "figma.com trello.com crates.io rubygems.org kaggle.com sourceforge.net").split()
ID_PREFIXES = "INC CHG PRJ OPS SEC FIN HR".split()
KEY_VALUES = ["max_workers=8", "TIMEOUT_MS=500", "retries=3", "cache_ttl=1h", "DEBUG=false", "pool_size=20",
              "batch_size=64", "region=eu-central-1"]
VERSIONS = ["v1.8.2", "v4.0.0-rc2", "v2.10.3", "v12.1", "release-2024.07", "v0.3.0-alpha", "build 4471"]
ELABORATE_KEY_VALUES = ["CACHE_MB=256", "workers=4", "timeout_ms=750", "RATE_LIMIT=100", "max_tokens=512",
                        "retry_delay=2s", "PORT=8081", "threads=16", "MAX_CONN=50", "ttl=300s", "SAMPLE_RATE=0.25",
                        "replicas=2"]
ELABORATE_COMMANDS = [
    "pnpm add -D vitest", "npx eslint .", "git fetch --all", "git reset --soft HEAD~1", "git tag -a v2", "kubectl describe pod",
    "kubectl apply -f k8s/", "kubectl scale deploy/api --replicas=3", "docker compose down -v", "docker logs -f worker",
    "npm run test:e2e", "yarn build", "pytest -k smoke", "alembic upgrade head", "redis-cli ping", "psql -U admin -d sales",
    "ssh deploy@bastion", "rsync -avz dist/ web:/srv", "tail -f app.log", "curl -I localhost:8080",
    "systemctl status postgres", "make lint", "poetry install", "dotnet build"]
# elaborate turns commands and KEY=VALUE into prose most: its exact-copy rows draw these kinds more often
EXTRA_KINDS = {"elaborate": ["command", "command", "command", "code_span", "key_value", "key_value", "version"]}
ELABORATE_VERSIONS = ["v3.2.1-beta", "v1.0.0-rc3", "v5.11", "v2.7.4", "v0.12.0", "v8.1.3-hotfix", "v4.4"]
NUMBERS = ["12500", "18750", "27000", "33400", "47500", "62000", "88200", "96000", "135000", "240000", "375000",
           "1250000", "2400000", "4100000", "00417", "00982", "000256", "0731", "7342", "6379", "8443", "9090",
           "27017", "15672", "641008", "400076", "110092", "600028", "1009", "7200"]

SHORTHAND = {"before": ["b4"], "tomorrow": ["tmrw", "tmr"], "account": ["acct"], "thanks": ["thx"],
             "because": ["cuz", "bc"], "please": ["pls", "plz"], "message": ["msg"], "meeting": ["mtg"],
             "manager": ["mgr"], "people": ["ppl"], "with": ["w/"], "without": ["w/o"], "you": ["u"], "your": ["ur"]}
COMMON = {"please", "with", "you", "your"}        # shorthand only next to a rarer one, never alone in a row
NOUN_ONLY = {"meeting"}                           # `mtg` is the noun: only after a determiner, never "meeting people"
DETERMINER = "(?:" + "|".join(rf"(?<=\b{d} )" for d in "a an the this that our my your next weekly team board".split()) + ")"


def sha_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pick(seq, key, shift=0):
    return seq[int(key[shift:shift + 8], 16) % len(seq)]


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def eval_strings():
    """(every `keep` string, every input) of eval_sets/: what training must not contain."""
    keeps, inputs = set(), set()
    for path in sorted(EVAL_SETS.rglob("*.json*")):
        data = read_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
        rows = data["rows"] if isinstance(data, dict) else data
        for r in rows:
            keeps.update(r.get("keep", []))
            inputs.add(r["input"].strip())
    return keeps, inputs


def leaks(token, keeps):
    """True when the token is an eval keep string, or one holds the other (short keeps like `5.3` aside)."""
    return any(token == k or (len(k) >= 8 and k in token) or (len(token) >= 8 and token in k) for k in keeps)


def token_pool(style=None):
    """{kind: sorted strings} that must come back byte for byte, most of them lowercase where a rewrite would capitalise."""
    style = style or STYLE
    pool = {
        "win_path": [f"C:\\Users\\{f}\\{d}\\{fl}" for f in FIRST for d in DIRS for fl in FILES[:4]]
                    + [f"D:\\{s}\\{e}\\{fl}" for s in SERVICES for e in ENVS for fl in FILES[4:]],
        "unix_path": [f"/home/{f}/{s}/{fl}" for f in FIRST for s in SERVICES[:3] for fl in FILES[4:]]
                     + [f"~/{s}/{e}/{fl}" for s in SERVICES for e in ENVS for fl in FILES[:4]]
                     + [f"/var/lib/{s}/{fl}" for s in SERVICES for fl in FILES],
        "command": list(COMMANDS),
        "code_span": [f"`{c}`" for c in COMMANDS],
        "host": [f"{s}.{e}.{d}" for s in SERVICES for e in ENVS for d in DOMAINS]
                + [f"{s}-0{i}.internal.{d}" for s in SERVICES for i in range(1, 4) for d in DOMAINS[:3]],
        "bare_domain": list(BARE_DOMAINS),
        "email": [f"{f}.{l}@{d}" for i, f in enumerate(FIRST) for j, l in enumerate(LAST) if (i + j) % 3 == 0
                  for d in DOMAINS[(i + j) % len(DOMAINS):][:1]]
                 + [f"{s}-team@{d}" for s in SERVICES for d in DOMAINS],
        "id": [f"{p}-{1000 + 37 * i}" for p in ID_PREFIXES for i in range(12)],
        "key_value": list(KEY_VALUES),
        "version": list(VERSIONS),
    }
    if style == "elaborate":
        pool["command"] += ELABORATE_COMMANDS
        pool["code_span"] += [f"`{c}`" for c in ELABORATE_COMMANDS]
        pool["key_value"] += ELABORATE_KEY_VALUES
        pool["version"] += ELABORATE_VERSIONS
    return {kind: sorted(set(tokens)) for kind, tokens in pool.items()}


def number_pool():
    return sorted(set(NUMBERS))


def clean_pool(keeps):
    pool = token_pool()
    bad = sorted(t for tokens in list(pool.values()) + [number_pool()] for t in tokens if leaks(t, keeps))
    if bad:
        raise SystemExit(f"token_pool() holds eval strings: {bad[:10]}")
    return pool


def single_link(r):
    """The row's one link when input and target each hold it exactly once and the row is a real rewrite."""
    urls = URL_RE.findall(r["input"])
    if len(urls) != 1 or r["input"].strip() == r["output"].strip():
        return None
    u = urls[0]
    return u if r["input"].count(u) == 1 and r["output"].count(u) == 1 else None


def opens_with_link(r):
    link = single_link(r)
    return bool(link) and r["input"].lstrip().startswith(link) and r["output"].lstrip().startswith(link)


def exact_copy_row(r, key, pool):
    link = single_link(r)
    kinds = sorted(pool) + EXTRA_KINDS.get(STYLE, [])
    token = pick(pool[pick(kinds, key)], key, 8)
    return {"row_index": r["row_index"], "input": r["input"].replace(link, token), "style": STYLE,
            "output": r["output"].replace(link, token), "cleanup": "KEEP"}


WORD = r"(?<![\w@/.\-`'\"#~\\]){}(?![\w@/.\-`'\"\\])"


def shorthand_pattern(word):
    return (DETERMINER if word in NOUN_ONLY else "") + WORD.format(word)


def shorthand_row(r, key):
    """Chat shorthand in the input for words the target also has; None when the row has no rare word to replace."""
    words = [w for w in SHORTHAND if re.search(shorthand_pattern(w), r["input"])
             and re.search(r"\b" + w + r"\b", r["output"], re.I)]
    if not any(w not in COMMON for w in words):
        return None
    text = r["input"]
    for n, w in enumerate(words):
        text = re.sub(shorthand_pattern(w), pick(SHORTHAND[w], key, 8 * (n % 7)), text, count=1)
    return {"row_index": r["row_index"], "input": text, "style": STYLE, "output": r["output"], "cleanup": "KEEP"}


def misspell(word, key):
    """A hard misspelling of `word` that is not a real word, or None: an inner vowel dropped, the last letter dropped,
    or two letters swapped."""
    vowels = [i for i, c in enumerate(word[1:-1], 1) if c in "aeiou"]
    options = []
    if vowels:
        options.append(word[:vowels[-1]] + word[vowels[-1] + 1:])          # broken -> brokn
        v = vowels[int(key[6:8], 16) % len(vowels)]
        options.append(word[:v] + word[v + 1:])                             # cancelled -> cancled
    options.append(word[:-1])                                               # update -> updat
    i = 1 + int(key[:4], 16) % (len(word) - 2)
    options.append(word[:i] + word[i + 1] + word[i] + word[i + 2:])         # dashboard -> dashbaord
    for typo in options[int(key[4:6], 16) % len(options):] + options:
        if typo != word and zipf_frequency(typo, "en") == 0:
            return typo
    return None


def typo_row(r, key):
    """Two or three hard misspellings in the input, of plain lowercase words the target has too."""
    words = sorted({w for w in re.findall(WORD.format(r"[a-z]{5,}"), r["input"])
                    if re.search(r"\b" + w + r"\b", r["output"])}, key=lambda w: sha_hex(key + w))
    edits = []
    for w in words:
        typo = misspell(w, sha_hex(key + w))
        if typo:
            edits.append((w, typo))
        if len(edits) == 2 + int(key[:2], 16) % 2:
            break
    if len(edits) < 2:
        return None
    text = r["input"]
    for w, typo in edits:
        text = re.sub(WORD.format(w), typo, text, count=1)
    return {"row_index": r["row_index"], "input": text, "style": STYLE, "output": r["output"], "cleanup": "KEEP"}


NUMBER = r"(?<![\w.,/:\-#@+]){}(?![\w,/:%\-]|\.\d)"


def plain_numbers(text):
    """Standalone numbers of 4-7 digits with no separator, not part of a date, time, version, id or decimal."""
    return re.findall(NUMBER.format(r"\d{4,7}"), text)


def number_row(r, key, numbers):
    """The row with one plain number (not a year), held once by input and target alike, replaced in both by a number
    of `numbers`; None when the row has none."""
    found = [n for n in sorted(set(plain_numbers(r["input"])))
             if not (len(n) == 4 and n[:2] in ("19", "20")) and plain_numbers(r["input"]).count(n) == 1
             and plain_numbers(r["output"]).count(n) == 1 and r["output"].count(n) == 1 and r["input"].count(n) == 1]
    if not found or r["input"].strip() == r["output"].strip():
        return None
    n, new = pick(found, key), pick(numbers, key, 8)
    pattern = NUMBER.format(n)
    return {"row_index": r["row_index"], "input": re.sub(pattern, new, r["input"], count=1), "style": STYLE,
            "output": re.sub(pattern, new, r["output"], count=1), "cleanup": "KEEP"}


def ranked(rows, salt):
    return sorted(rows, key=lambda r: sha_hex(salt + r["input"]))


def make_rows(candidates, wanted, held, salt, build):
    """(train rows, held-out rows, base rows of the held-out ones): the first rows `build` accepts in hash order,
    held-out rows first. The bases are returned so that no replay row shows a held-out target."""
    train, heldout, bases = [], [], []
    for r in ranked(candidates, salt):
        row = build(r, sha_hex(salt + r["input"]))
        if row is None:
            continue
        if len(heldout) < held:
            heldout.append(row)
            bases.append(r)
        else:
            train.append(row)
        if len(train) == wanted:
            break
    return train, heldout, bases


def build(keeps, eval_inputs, out_root=PHASE3):
    cfg = STYLES[STYLE]
    exact_rows, shorthand_rows, typo_rows, number_rows = (cfg[k] for k in ("exact_copy", "shorthand", "typos", "numbers"))
    heldout_rows = cfg["heldout"]
    src = PHASE2 / STYLE
    print(verify_manifest.verify(src), flush=True)
    pool = clean_pool(keeps)
    p2_evals = {r["input"].strip() for f in sorted((src / "evals").glob("*.jsonl")) for r in read_jsonl(f)}
    banned = p2_evals | eval_inputs

    def usable(path):
        return [r for r in read_jsonl(path) if r["input"].strip() not in banned and r["input"].strip() != r["output"].strip()
                and r.get("style") == STYLE and r.get("cleanup", "KEEP") == "KEEP"]

    url_rows = usable(src / "train" / f"phase_2_part_02_{STYLE}.jsonl")
    date_rows = usable(src / "train" / f"phase_2_part_01_{STYLE}.jsonl")
    augment = usable(src / "train" / "phase_2_augment_tokens.jsonl")
    p1_rows = usable(src / "train" / "phase_1_dataset.jsonl")

    # The token opening the text is where a rewrite capitalises most, and such rows are few: all of them first.
    linked = [r for r in url_rows if single_link(r)]
    first = [r for r in linked if opens_with_link(r)]
    held_first = round(heldout_rows["exact_copy"] * len(first) / len(linked))
    exact, exact_held, exact_bases = make_rows(first, len(first), held_first, f"{SALT}-exact",
                                               lambda r, k: exact_copy_row(r, k, pool))
    rest = make_rows([r for r in linked if not opens_with_link(r)], exact_rows - len(exact),
                     heldout_rows["exact_copy"] - held_first, f"{SALT}-exact", lambda r, k: exact_copy_row(r, k, pool))
    exact, exact_held, exact_bases = exact + rest[0], exact_held + rest[1], exact_bases + rest[2]
    used = {r["row_index"] for r in exact + exact_held}
    casual = [r for r in date_rows + url_rows if r["row_index"] not in used]
    nums, num_held, num_bases = [], [], []
    if number_rows:
        nums, num_held, num_bases = make_rows(casual + p1_rows, number_rows, heldout_rows["numbers"], f"{SALT}-number",
                                              lambda r, k: number_row(r, k, number_pool()))
        taken = {(r["row_index"], r["output"]) for r in num_bases} | {(r["row_index"], r["input"]) for r in nums}
        casual = [r for r in casual if (r["row_index"], r["output"]) not in taken]
        p1_rows = [r for r in p1_rows if (r["row_index"], r["output"]) not in taken]
    short, short_held, short_bases = make_rows(casual + p1_rows, shorthand_rows, heldout_rows["shorthand"], f"{SALT}-short", shorthand_row)
    used_bases = {(r["row_index"], r["output"]) for r in short + short_held}     # shorthand keeps both of its base
    typo_base = [r for r in casual + p1_rows if (r["row_index"], r["output"]) not in used_bases]
    typos, typo_held, typo_bases = make_rows(typo_base, typo_rows, heldout_rows["typos"], f"{SALT}-typo", typo_row)

    skip = ({r["input"].strip() for r in exact + nums + short + typos + exact_held + num_held + short_held + typo_held}
            | {r["input"].strip() for r in exact_bases + num_bases + short_bases + typo_bases})
    p2_replay = [r for r in ranked(date_rows + url_rows + augment, f"{SALT}-p2replay")
                 if r["input"].strip() not in skip][:PHASE2_REPLAY]
    p1_replay = [r for r in ranked(p1_rows, f"{SALT}-p1replay") if r["input"].strip() not in skip][:PHASE1_REPLAY]
    for name, got, wanted in (("exact_copy", exact, exact_rows), ("numbers", nums, number_rows), ("shorthand", short, shorthand_rows),
                              ("typos", typos, typo_rows), ("phase_2_replay", p2_replay, PHASE2_REPLAY),
                              ("phase_1_replay", p1_replay, PHASE1_REPLAY)):
        assert len(got) == wanted, (name, len(got), wanted)

    out = Path(out_root) / STYLE
    files = {"train/phase_3_exact_copy.jsonl": exact}
    if number_rows:
        files["train/phase_3_numbers.jsonl"] = nums
    files.update({
        "train/phase_3_shorthand.jsonl": short, "train/phase_3_typos.jsonl": typos,
        "train/phase_2_replay.jsonl": p2_replay, "train/phase_1_replay.jsonl": p1_replay,
        "evals/phase_3_heldout.jsonl": exact_held + num_held + short_held + typo_held,
        "evals/phase_2_heldout.jsonl": read_jsonl(src / "evals" / "phase_2_heldout.jsonl"),
        "evals/phase_1_eval.jsonl": read_jsonl(src / "evals" / "phase_1_eval.jsonl"),
    })
    for name, rows in files.items():
        write_jsonl(out / name, rows)

    def entry(name):
        path = out / name
        return {"rows": len(files[name]), "sha256": verify_manifest.file_sha256(path)}

    manifest = {
        "style": STYLE,
        "built_by": "SFT_adapter_rephrase/data_prep/build_phase_3_dataset.py",
        "train_files": {n: entry(n) for n in files if n.startswith("train/")},
        "eval_files": {n: entry(n) for n in files if n.startswith("evals/")},
        "counts": {"phase_3_exact_copy": len(exact), **({"phase_3_numbers": len(nums)} if number_rows else {}),
                   "phase_3_shorthand": len(short), "phase_3_typos": len(typos),
                   "phase_2_replay": len(p2_replay), "phase_1_replay": len(p1_replay),
                   "phase_3_heldout": len(files["evals/phase_3_heldout.jsonl"]),
                   "phase_2_heldout": len(files["evals/phase_2_heldout.jsonl"]),
                   "phase_1_eval": len(files["evals/phase_1_eval.jsonl"])},
        "selection": {
            "base_rows": f"rows of {src}/train with style {STYLE!r}, cleanup KEEP, input != output, input not in "
                         f"{src}/evals nor in any eval_sets/ file; ranked by sha256('{SALT}-<kind>' + input). The first "
                         f"accepted rows of each kind are held out (evals/phase_3_heldout.jsonl), the next ones trained on.",
            "exact_copy": "phase-2 url rows whose input and target each hold their single link once; the link is replaced "
                          "in both by a token of token_pool(): kind and token picked by the row's hash. Every row whose "
                          "input and target open with the link is used (held-out share in proportion), then the others.",
            **({"numbers": "date and url rows not used above and phase-1 replay rows whose input and target each hold "
                           "the same plain number (4-7 digits, no separator, not a year 19xx/20xx) once; it is replaced "
                           "in both by a number of number_pool() picked by the row's hash"} if number_rows else {}),
            "shorthand": f"date and url rows not used above and phase-1 replay rows; the first occurrence in the input of each of {sorted(SHORTHAND)} "
                         f"that the target also has becomes its shorthand (a row needs one word outside {sorted(COMMON)}).",
            "typos": "rows not used above, phase-1 replay included; 2-3 lowercase words of 5+ letters that the target also "
                     "has get a hard misspelling (an inner vowel dropped, last letter dropped or two letters swapped) "
                     "with wordfreq zipf 0 (not a word).",
            "phase_2_replay": f"the first {PHASE2_REPLAY} rows of both phase-2 parts and the augment file by "
                              f"sha256('{SALT}-p2replay' + input), new-row inputs and the bases of held-out rows excluded.",
            "phase_1_replay": f"the first {PHASE1_REPLAY} rows of the phase-2 folder's phase_1_dataset.jsonl by "
                              f"sha256('{SALT}-p1replay' + input), new-row inputs and the bases of held-out rows excluded.",
            "evals": "phase_2_heldout and phase_1_eval are byte-identical copies of the phase-2 eval files.",
        },
        "token_pool": {kind: len(tokens) for kind, tokens in pool.items()},
        **({"number_pool": len(number_pool())} if number_rows else {}),
        "sources": {"phase_2_manifest_sha256": verify_manifest.file_sha256(src / "manifest.json"),
                    "eval_sets_keep_strings_checked": len(keeps)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(STYLE, json.dumps(manifest["counts"]), flush=True)
    print(verify_manifest.verify(out), flush=True)


def main():
    global STYLE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--style", default="professional", choices=sorted(STYLES))
    ap.add_argument("--out-root", type=Path, default=PHASE3, help=f"default {PHASE3}; a style gets <out-root>/<style>")
    args = ap.parse_args()
    STYLE = args.style
    build(*eval_strings(), out_root=args.out_root)


if __name__ == "__main__":
    main()
