"""Freeze the phase-3 training set of an adapter (professional, elaborate, concise): a short repair pass on top of phase 2.

    /root/venv/bin/python SFT_adapter_rephrase/data_prep/build_phase_3_dataset.py                    # professional
    /root/venv/bin/python SFT_adapter_rephrase/data_prep/build_phase_3_dataset.py --style elaborate
    /root/venv/bin/python SFT_adapter_rephrase/data_prep/build_phase_3_dataset.py --style concise

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

Concise gets the three kinds of professional, sized for its own habits: it capitalises a copied string that opens
the text (`j.smith@...` -> `J.Smith@`, `docker compose` -> `Docker compose`), shortens commands (`kubectl` -> `K8s`,
`git rebase -i` loses `-i`), drops or misreads shorthand (`wfh` -> "what", `idk`, `rn`, `nvm` dropped) and short
misspelled key words (`cancled` -> "closed", `refnd` -> "repair"). So its exact-copy rows add CONCISE_COMMANDS
(commands with flags) and draw commands, code spans, emails and hosts more often, and its shorthand rows also use
CONCISE_SHORTHAND (wfh, idk, rn, nvm, asap, btw, info, appt). The first pass (v1) still capitalised a host or
version opening the text, read `tmrw`/`mtg`/`wfh` as letters and lost `idk`/`rn`/`nvm` (the phase-2 rows hold them
less than 10 times each), and misread short key words in short messages, so concise also gets

  train/phase_3_chat.jsonl         short chat messages from CONCISE_CHAT, slots filled from CHAT_SLOTS: the input
                                   holds the shorthand, the target the plain words (written for this set, none of
                                   them an eval_sets/ sentence)
  train/phase_3_short_typos.jsonl  rows of at most 25 words with one or two misspelled words of 4+ letters
                                   that the target also has
  more exact-copy rows             every row opening with its link is used START_REPEAT times, each time with a token
                                   of START_KINDS (host, domain, email, id, version, command, KEY=VALUE)
  train/phase_3_key_typos.jsonl    (v3) v2 still read `cancled` as "closed" and `refnd` as "friend": rows of at most
                                   40 words holding one of KEY_WORDS (everyday message words) get it misspelled
  train/phase_3_word_chat.jsonl    (v4) v3 still did: short messages from TYPO_CHAT, where the key word ({w}) is
                                   misspelled by misspell() in a few ways and the target spells it right
  quoted commands                  (v3) the command kind also comes in “curly quotes”, and pnpm commands are added

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
    "concise": {"exact_copy": 3600, "numbers": 0, "shorthand": 2200, "typos": 1800, "chat": 1000, "short_typos": 1400,
                "key_typos": 1150, "word_chat": 1200, "replay": 5600,
                "heldout": {"exact_copy": 150, "shorthand": 75, "typos": 75, "chat": 50, "short_typos": 50, "key_typos": 50,
                            "word_chat": 50}},
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
EXTRA_KINDS = {"elaborate": ["command", "command", "command", "code_span", "key_value", "key_value", "version"],
               "concise": ["command", "command", "command", "code_span", "email", "host"]}
# concise cuts flags and renames tools (`kubectl` -> `K8s`): commands whose flags must stay
CONCISE_COMMANDS = [
    "git rebase --continue", "git commit --amend --no-edit", "git push -u origin feat/search", "git cherry-pick -n 9f8e7d6",
    "kubectl rollout status deploy/cart", "kubectl get svc -A", "kubectl port-forward svc/db 5432", "kubectl delete pod -l app=cron",
    "docker compose pull", "docker run --rm -it alpine sh", "docker exec -it db psql", "docker system prune -af",
    "npm i -g pnpm", "npm run build -- --watch", "npm ls react", "pip install -e .", "pip freeze", "python3 -m venv .venv",
    "sed -i s/foo/bar/ conf.ini", "find . -name '*.log' -delete", "du -sh *", "journalctl -u nginx -f", "ps aux",
    "terraform init -upgrade", "ansible-playbook site.yml --check", "az login", "ssh -L 9000:localhost:9000 dev",
    "pnpm add zod", "pnpm test", "pnpm dlx prisma migrate", "pnpm run dev", "npx tsc --noEmit"]
ELABORATE_VERSIONS = ["v3.2.1-beta", "v1.0.0-rc3", "v5.11", "v2.7.4", "v0.12.0", "v8.1.3-hotfix", "v4.4"]
NUMBERS = ["12500", "18750", "27000", "33400", "47500", "62000", "88200", "96000", "135000", "240000", "375000",
           "1250000", "2400000", "4100000", "00417", "00982", "000256", "0731", "7342", "6379", "8443", "9090",
           "27017", "15672", "641008", "400076", "110092", "600028", "1009", "7200"]

SHORTHAND = {"before": ["b4"], "tomorrow": ["tmrw", "tmr"], "account": ["acct"], "thanks": ["thx"],
             "because": ["cuz", "bc"], "please": ["pls", "plz"], "message": ["msg"], "meeting": ["mtg"],
             "manager": ["mgr"], "people": ["ppl"], "with": ["w/"], "without": ["w/o"], "you": ["u"], "your": ["ur"]}
COMMON = {"please", "with", "you", "your"}        # shorthand only next to a rarer one, never alone in a row
CONCISE_SHORTHAND = {"working from home": ["wfh"], "I don't know": ["idk"], "right now": ["rn"],
                     "never mind": ["nvm"], "as soon as possible": ["asap"], "by the way": ["btw"],
                     "information": ["info"], "appointment": ["appt"]}
# (input, target) templates; {thing} {person} {team} {day} from CHAT_SLOTS, a capitalised slot name capitalises it
CONCISE_CHAT = [
    ("{person} said the {thing} will be ready tmrw", "{Person} said the {thing} will be ready tomorrow."),
    ("can we push the call to tmrw morning? {person} is out", "Can we move the call to tomorrow morning? {Person} is out."),
    ("tmrw i'll send the {thing} first thing", "I'll send the {thing} first thing tomorrow."),
    ("is tmrw still ok for the {team} review", "Is tomorrow still OK for the {team} review?"),
    ("off tmrw, {person} covers for me", "I'm off tomorrow; {person} is covering for me."),
    ("pls remind me tmrw about the {thing}", "Please remind me about the {thing} tomorrow."),
    ("tmr is the deadline for the {thing}", "Tomorrow is the deadline for the {thing}."),
    ("tmrw suits {person} better for the {thing}", "Tomorrow suits {person} better for the {thing}."),
    ("tmrw afternoon is free for the {team} call", "Tomorrow afternoon is free for the {team} call."),
    ("tmrw could work if {person} signs the {thing}", "Tomorrow could work if {person} signs the {thing}."),
    ("mtg w/ {team} is at 3, dont be late", "The meeting with {team} is at 3. Don't be late."),
    ("can u move our mtg to {day}", "Can you move our meeting to {Day}?"),
    ("the {team} mtg got cancelled", "The {team} meeting was cancelled."),
    ("pls send notes from the mtg about the {thing}", "Please send the notes from the meeting about the {thing}."),
    ("running late for the mtg, start without me", "I'm running late for the meeting. Start without me."),
    ("next mtg is {day}, bring the {thing}", "The next meeting is {Day}. Bring the {thing}."),
    ("cant talk rn, driving to {person}", "I can't talk right now; I'm driving to {person}."),
    ("the {thing} is being checked rn", "The {thing} is being checked right now."),
    ("{person} is busy rn, try after lunch", "{Person} is busy right now. Try after lunch."),
    ("rn the {thing} is stuck with {team}", "Right now, the {thing} is stuck with {team}."),
    ("im swamped rn, can the {thing} wait till {day}", "I'm swamped right now. Can the {thing} wait until {Day}?"),
    ("idk who approved the {thing}", "I don't know who approved the {thing}."),
    ("idk when {person} will reply about the {thing}", "I don't know when {person} will reply about the {thing}."),
    ("idk, ask {person} about the {thing}", "I'm not sure. Ask {person} about the {thing}."),
    ("honestly idk where the {thing} went", "Honestly, I don't know where the {thing} went."),
    ("idk why {team} rejected the {thing}", "I don't know why {team} rejected the {thing}."),
    ("nvm, {person} already sent the {thing}", "Never mind, {person} already sent the {thing}."),
    ("nvm the {thing}, its sorted", "Never mind the {thing}; it's sorted."),
    ("oh nvm, {person} fixed it", "Oh, never mind. {Person} fixed it."),
    ("nvm about {day}, {team} can do it", "Never mind about {Day}; {team} can do it."),
    ("wfh on {day}, call my mobile if the {thing} comes", "I'm working from home on {Day}; call my mobile if the {thing} comes."),
    ("{person} is wfh this week", "{Person} is working from home this week."),
    ("can i wfh {day}? the {thing} is almost done", "Can I work from home on {Day}? The {thing} is almost done."),
    ("most of {team} is wfh on {day}", "Most of {team} is working from home on {Day}."),
    ("wfh today bc the trains are down, {person} has the {thing}", "I'm working from home today because the trains are down; {person} has the {thing}."),
    ("send the {thing} b4 {day} pls", "Please send the {thing} before {Day}."),
    ("check w/ {person} b4 paying the {thing}", "Check with {person} before paying the {thing}."),
    ("my acct got locked again, {team} pls help", "My account is locked again. {Team}, please help."),
    ("{person} needs access to the {team} acct", "{Person} needs access to the {team} account."),
]
# (key word, input, target): {w} is the key word, misspelled in the input
TYPO_CHAT = [
    ("refund", "{person} asked for a {w} on the {thing}", "{Person} asked for a refund on the {thing}."),
    ("refund", "still waiting on my {w} from {day}", "Still waiting on my refund from {Day}."),
    ("refund", "can u process the {w} today pls", "Can you process the refund today, please?"),
    ("refund", "the {w} for the {thing} went to the wrong card", "The refund for the {thing} went to the wrong card."),
    ("cancelled", "{person} {w} the {thing} last night", "{Person} cancelled the {thing} last night."),
    ("cancelled", "the {team} call is {w}, see u {day}", "The {team} call is cancelled. See you {Day}."),
    ("cancelled", "my booking got {w} without notice", "My booking was cancelled without notice."),
    ("cancelled", "they {w} the {thing} bc of the price", "They cancelled the {thing} because of the price."),
    ("canceled", "{person} {w} our subscription", "{Person} canceled our subscription."),
    ("still", "the {thing} is {w} missing a signature", "The {thing} is still missing a signature."),
    ("still", "{person} is {w} on leave till {day}", "{Person} is still on leave until {Day}."),
    ("still", "are we {w} on for {day}", "Are we still on for {Day}?"),
    ("until", "the office is closed {w} {day}", "The office is closed until {Day}."),
    ("until", "hold the {thing} {w} {person} signs", "Hold the {thing} until {person} signs."),
    ("broken", "the {thing} link is {w} again", "The {thing} link is broken again."),
    ("broken", "my laptop charger is {w}, {team} pls send one", "My laptop charger is broken. {Team}, please send one."),
    ("delivered", "the {thing} was {w} to the wrong office", "The {thing} was delivered to the wrong office."),
    ("delivery", "{w} is late again, {person} is annoyed", "The delivery is late again; {person} is annoyed."),
    ("invoice", "pls resend the {w} for {day}", "Please resend the invoice for {Day}."),
    ("receipt", "lost the {w} for the {thing}", "I lost the receipt for the {thing}."),
    ("payment", "the {w} for the {thing} bounced", "The payment for the {thing} bounced."),
    ("password", "forgot my {w} again, {team} can u reset it", "I forgot my password again. {Team}, can you reset it?"),
    ("approved", "{person} {w} the {thing} this morning", "{Person} approved the {thing} this morning."),
    ("confirmed", "{person} {w} the {thing} for {day}", "{Person} confirmed the {thing} for {Day}."),
    ("delayed", "the {thing} is {w} till {day}", "The {thing} is delayed until {Day}."),
    ("damaged", "the {thing} arrived {w}", "The {thing} arrived damaged."),
    ("missing", "page 3 of the {thing} is {w}", "Page 3 of the {thing} is missing."),
    ("returned", "{person} {w} the {thing} yesterday", "{Person} returned the {thing} yesterday."),
    ("replace", "can {team} {w} the {thing} by {day}", "Can {team} replace the {thing} by {Day}?"),
    ("crashed", "the app {w} when i opened the {thing}", "The app crashed when I opened the {thing}."),
    ("failed", "the upload {w} twice, {person} pls check", "The upload failed twice. {Person}, please check."),
    ("schedule", "can u send the {team} {w} for {day}", "Can you send the {team} schedule for {Day}?"),
    ("deadline", "the {w} for the {thing} moved to {day}", "The deadline for the {thing} moved to {Day}."),
    ("urgent", "{w}: {person} needs the {thing} now", "Urgent: {person} needs the {thing} now."),
]
CHAT_SLOTS = {"thing": "invoice contract payment report slides quote budget order booking draft".split(),
              "person": ["priya", "the vendor", "my manager", "the landlord", "IT", "the courier", "sam", "the client"],
              "team": "sales legal design HR ops marketing".split(),
              "day": "monday tuesday wednesday thursday friday".split()}
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
    if style == "concise":
        pool["command"] += CONCISE_COMMANDS
        pool["code_span"] += [f"`{c}`" for c in CONCISE_COMMANDS] + [f"“{c}”" for c in COMMANDS + CONCISE_COMMANDS]
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


def shorthand_words():
    return {**SHORTHAND, **CONCISE_SHORTHAND} if STYLE == "concise" else SHORTHAND


def shorthand_row(r, key):
    """Chat shorthand in the input for words the target also has; None when the row has no rare word to replace."""
    table = shorthand_words()
    words = [w for w in table if re.search(shorthand_pattern(w), r["input"])
             and re.search(r"\b" + w + r"\b", r["output"], re.I)]
    if not any(w not in COMMON for w in words):
        return None
    text = r["input"]
    for n, w in enumerate(words):
        text = re.sub(shorthand_pattern(w), pick(table[w], key, 8 * (n % 7)), text, count=1)
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


def short_typo_row(r, key):
    """One or two misspellings in an input of at most 25 words, of lowercase words of 4+ letters the target has too."""
    if len(r["input"].split()) > 25:
        return None
    words = sorted({w for w in re.findall(WORD.format(r"[a-z]{4,}"), r["input"])
                    if re.search(r"\b" + w + r"\b", r["output"])}, key=lambda w: sha_hex(key + w))
    edits = []
    for w in words:
        typo = misspell(w, sha_hex(key + w))
        if typo:
            edits.append((w, typo))
        if len(edits) == 1 + int(key[:2], 16) % 2:
            break
    if not edits:
        return None
    text = r["input"]
    for w, typo in edits:
        text = re.sub(WORD.format(w), typo, text, count=1)
    return {"row_index": r["row_index"], "input": text, "style": STYLE, "output": r["output"], "cleanup": "KEEP"}


def fill(template, slots):
    return template.format(**slots, **{k.capitalize(): (v[:1].upper() + v[1:]) for k, v in slots.items()})


def chat_rows():
    """Every CONCISE_CHAT template with every slot combination, as base rows (row_index -1: not from phase 2)."""
    rows = []
    for t, (inp, out) in enumerate(CONCISE_CHAT):
        names = sorted(set(re.findall(r"{(\w+)}", inp + out.lower())))
        combos = [{}]
        for n in names:
            combos = [dict(c, **{n: v}) for c in combos for v in CHAT_SLOTS[n]]
        rows += [{"row_index": -1 - t, "input": fill(inp, c), "style": STYLE, "output": fill(out, c), "cleanup": "KEEP"}
                 for c in combos]
    return rows


def word_chat_rows():
    """TYPO_CHAT with every slot combination and the key word misspelled (the misspelling picked by the filled text)."""
    rows = []
    for t, (word, inp, out) in enumerate(TYPO_CHAT):
        names = sorted(set(re.findall(r"{(\w+)}", inp.lower() + out.lower())) - {"w"})
        combos = [{}]
        for n in names:
            combos = [dict(c, **{n: v}) for c in combos for v in CHAT_SLOTS[n]]
        for c in combos:
            text = fill(inp, dict(c, w="{w}"))
            typo = misspell(word, sha_hex(text))
            if typo:
                rows.append({"row_index": -101 - t, "input": text.replace("{w}", typo), "style": STYLE,
                             "output": fill(out, c), "cleanup": "KEEP"})
    return rows


START_KINDS = ["host", "bare_domain", "email", "id", "version", "command", "key_value", "version"]
START_REPEAT = 4
KEY_WORDS = ("cancelled canceled refund refunds broken still until delivery deliver order invoice receipt payment paid "
             "account password booking booked meeting schedule delayed shipped package address replace return "
             "returned arrived damaged missing urgent confirm confirmed approve approved budget contract deadline "
             "update upload download error crashed failed working request").split()


def key_typo_row(r, key):
    """One KEY_WORDS word misspelled in an input of at most 40 words whose target has the word too."""
    if len(r["input"].split()) > 40:
        return None
    words = sorted((w for w in KEY_WORDS if re.search(WORD.format(w), r["input"]) and re.search(r"\b" + w + r"\b", r["output"], re.I)),
                   key=lambda w: sha_hex(key + w))
    for w in words:
        typo = misspell(w, sha_hex(key + w))
        if typo:
            return {"row_index": r["row_index"], "input": re.sub(WORD.format(w), typo, r["input"], count=1),
                    "style": STYLE, "output": r["output"], "cleanup": "KEEP"}
    return None


def start_copy_row(r, key, pool):
    link = single_link(r)
    token = pick(pool[pick(START_KINDS, key)], key, 8)
    return {"row_index": r["row_index"], "input": r["input"].replace(link, token), "style": STYLE,
            "output": r["output"].replace(link, token), "cleanup": "KEEP"}


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
    if cfg.get("chat"):     # concise: the opening rows again, with start-prone tokens
        held_inputs = {r["input"] for r in exact_bases}
        for n in range(1, START_REPEAT):
            exact += [start_copy_row(r, sha_hex(f"{SALT}-start{n}" + r["input"]), pool)
                      for r in ranked(first, f"{SALT}-start{n}") if r["input"] not in held_inputs]
        exact = exact[:exact_rows]
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
    chat, chat_held, styps, styp_held, styp_bases, ktyps, wchat = [], [], [], [], [], [], []
    if cfg.get("chat"):
        chat, chat_held, _ = make_rows(chat_rows(), cfg["chat"], heldout_rows["chat"], f"{SALT}-chat", lambda r, k: r)
        taken = used_bases | {(r["row_index"], r["output"]) for r in typos + typo_bases}
        styps, styp_held, styp_bases = make_rows([r for r in typo_base if (r["row_index"], r["output"]) not in taken],
                                                 cfg["short_typos"], heldout_rows["short_typos"], f"{SALT}-styp", short_typo_row)
        taken |= {(r["row_index"], r["output"]) for r in styps + styp_bases}
        ktyps, ktyp_held, ktyp_bases = make_rows([r for r in typo_base if (r["row_index"], r["output"]) not in taken],
                                                 cfg["key_typos"], heldout_rows["key_typos"], f"{SALT}-ktyp", key_typo_row)
        styps, styp_held, styp_bases = styps, styp_held + ktyp_held, styp_bases + ktyp_bases
        wchat, wchat_held, _ = make_rows(word_chat_rows(), cfg["word_chat"], heldout_rows["word_chat"], f"{SALT}-wchat",
                                         lambda r, k: r)
        chat, chat_held = chat + wchat, chat_held + wchat_held
    replay_p2 = replay_p1 = cfg.get("replay", PHASE2_REPLAY)

    skip = ({r["input"].strip() for r in exact + nums + short + typos + exact_held + num_held + short_held + typo_held
             + styps + styp_held + ktyps}
            | {r["input"].strip() for r in exact_bases + num_bases + short_bases + typo_bases + styp_bases})
    p2_replay = [r for r in ranked(date_rows + url_rows + augment, f"{SALT}-p2replay")
                 if r["input"].strip() not in skip][:replay_p2]
    p1_replay = [r for r in ranked(p1_rows, f"{SALT}-p1replay") if r["input"].strip() not in skip][:replay_p1]
    for name, got, wanted in (("exact_copy", exact, exact_rows), ("numbers", nums, number_rows), ("shorthand", short, shorthand_rows),
                              ("typos", typos, typo_rows),
                              ("chat", chat, cfg.get("chat", 0) + cfg.get("word_chat", 0)), ("short_typos", styps, cfg.get("short_typos", 0)),
                              ("key_typos", ktyps, cfg.get("key_typos", 0)),
                              ("phase_2_replay", p2_replay, replay_p2), ("phase_1_replay", p1_replay, replay_p1)):
        assert len(got) == wanted, (name, len(got), wanted)

    out = Path(out_root) / STYLE
    files = {"train/phase_3_exact_copy.jsonl": exact}
    if number_rows:
        files["train/phase_3_numbers.jsonl"] = nums
    files.update({
        "train/phase_3_shorthand.jsonl": short, "train/phase_3_typos.jsonl": typos,
        **({"train/phase_3_chat.jsonl": chat[:len(chat) - len(wchat)],
            "train/phase_3_word_chat.jsonl": wchat, "train/phase_3_short_typos.jsonl": styps,
            "train/phase_3_key_typos.jsonl": ktyps} if chat else {}),
        "train/phase_2_replay.jsonl": p2_replay, "train/phase_1_replay.jsonl": p1_replay,
        "evals/phase_3_heldout.jsonl": exact_held + num_held + short_held + typo_held + chat_held + styp_held,
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
                   **({"phase_3_chat": len(chat) - len(wchat), "phase_3_word_chat": len(wchat), "phase_3_short_typos": len(styps),
                       "phase_3_key_typos": len(ktyps)} if chat else {}),
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
            "shorthand": f"date and url rows not used above and phase-1 replay rows; the first occurrence in the input of each of {sorted(shorthand_words())} "
                         f"that the target also has becomes its shorthand (a row needs one word outside {sorted(COMMON)}).",
            "typos": "rows not used above, phase-1 replay included; 2-3 lowercase words of 5+ letters that the target also "
                     "has get a hard misspelling (an inner vowel dropped, last letter dropped or two letters swapped) "
                     "with wordfreq zipf 0 (not a word).",
            **({"chat": f"{len(CONCISE_CHAT)} CONCISE_CHAT templates x CHAT_SLOTS combinations, ranked by "
                        f"sha256('{SALT}-chat' + input)",
                "short_typos": "rows not used above of at most 25 words; 1-2 lowercase words of 4+ letters the target "
                               "also has get a hard misspelling (as typos)",
                "word_chat": f"{len(TYPO_CHAT)} TYPO_CHAT templates x CHAT_SLOTS, key word misspelled by misspell(), "
                             f"ranked by sha256('{SALT}-wchat' + input)",
                "key_typos": "rows not used above of at most 40 words; one KEY_WORDS word the target also has gets a hard "
                             "misspelling",
                "start_repeat": f"rows opening with their link are used {START_REPEAT} times, the extra times with a "
                                f"token of {START_KINDS}"} if chat else {}),
            "phase_2_replay": f"the first {replay_p2} rows of both phase-2 parts and the augment file by "
                              f"sha256('{SALT}-p2replay' + input), new-row inputs and the bases of held-out rows excluded.",
            "phase_1_replay": f"the first {replay_p1} rows of the phase-2 folder's phase_1_dataset.jsonl by "
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
