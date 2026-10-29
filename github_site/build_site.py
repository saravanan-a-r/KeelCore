"""Turn the generated report page into the static site.

usage: python build_site.py path/to/report.html path/to/rewrites.json

The report page is produced by the report generator (every number comes from the run's own
logs and evaluation records). This script only changes the presentation: it drops the product
name, self-hosts the charts library and fonts, and adds the site chrome (top bar, contents
rail, theme switch, reading progress). Run it again whenever the report is regenerated.

`rewrites.json` holds the wording fixes that are specific to one generated report, so it is kept
next to that report rather than here: `replace` is a list of [old, new] pairs, each of which must
occur in the report, and `forbidden` is a regular expression that must not match the finished page.
"""
import datetime as dt
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if len(sys.argv) != 3:
    sys.exit(__doc__)
src = Path(sys.argv[1]).read_text()
REWRITES = json.loads(Path(sys.argv[2]).read_text())
FORBIDDEN = re.compile(REWRITES["forbidden"], re.I)

TITLE = "Pretraining a 752M Encoder–Decoder for Text Rewriting"
DESC = ("Technical report: a 752-million-parameter T5-style encoder–decoder pretrained on one GPU over six weeks, "
        "with full training dynamics, evaluation during training, engineering checks and a comparison with open models of similar size.")

# ---- 1. neutral wording (no product name anywhere) -------------------------------------------
src = re.sub(r"(?:KeelCore|EchoMeBetter)(?:-752M)?", "the 752M model", src)
# sentence starts (after a tag, a full stop or an opening quote) take a capital
src = re.sub(r'(^|>|\. |"|: )the 752M model', lambda m: m.group(1) + "The 752M model", src)
src = src.replace("<footer>The 752M model technical report.", "<footer>Technical report.")
body_start = src.index('<div class="shell">')
head_src = src[:body_start]
body = src[body_start:]
style = re.search(r"<style>.*?</style>", head_src, re.S).group(0)

# h1: the product name was the title
body = re.sub(r"<h1>.*?</h1>", f"<h1>{TITLE}</h1>", body, count=1)
body = body.replace('<div class="kicker">Technical report · Pretraining</div>', '<div class="kicker">Technical report · Pretraining</div>', 1)

# ---- 2. local assets ------------------------------------------------------------------------
body = body.replace("https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js", "assets/chart.umd.min.js")

# ---- 3. site chrome ----------------------------------------------------------------------------
bar = f'''<header class="topbar" role="banner">
  <button class="btn" id="menuBtn" type="button" aria-expanded="false" aria-controls="toc"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 4h12M2 8h12M2 12h12"/></svg><span>Contents</span></button>
  <div class="name">Technical report<em>752M encoder–decoder</em></div>
  <div class="sp"></div>
  <div class="now" id="now" aria-live="off"></div>
  <button class="btn" id="themeBtn" type="button"><svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="3.2"/><path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1"/></svg><span>Dark</span></button>
  <div class="progress" id="progress"></div>
</header>
'''

# ---- data: only the commercially usable datasets, with their token budget ------------------------
# The pretraining datasets and their token budget. The data section, its chart and the chart data
# are all built from this list.
BUDGET = [  # label, license, billions of raw corpus tokens
    ("Educational web (FineWeb-Edu)", "ODC-By 1.0", 39.854),
    ("General web (C4)", "ODC-By 1.0", 10.619),
    ("Synthetic textbooks (Cosmopedia)", "Apache-2.0", 7.544),
    ("Permissively licensed code", "MIT / Apache-2.0 / BSD-3", 3.366),
    ("Public-domain books (PG-19)", "Apache-2.0", 1.665),
    ("tldr pages", "CC BY 4.0", 0.003),
    ("Command-line cheat sheets", "CC0 1.0", 0.0002),
]
DATA_SECTION = f"""<section id="data">
  <h2><span class="n">2</span>Data</h2>
  <p>The pretraining data comes from seven public datasets, all under licenses that allow commercial use: educational and general web text, synthetic textbooks, permissively licensed source code, public-domain books and two small command-line references. The token budget is 63.05 billion raw tokens under the model's own tokenizer, which is 68.48 billion processed tokens once the sentinel tokens and objective formatting are added. The SentencePiece tokenizer reserves 896 pieces for command-line vocabulary such as flags and utilities.</p>
  <figure class="fig"><div class="frame"><div class="chart tall"><canvas id="cCorpus" role="img" aria-label="Token budget by dataset"></canvas></div></div>
    <figcaption><b>Figure 1.</b> Token budget by dataset, in billions of raw tokens under the model's own tokenizer.</figcaption></figure>
  <h3>How the data was read</h3>
  <p>Files were read one at a time in a fixed shuffled order, and each file holds a single dataset, so training saw blocks of consecutive steps from one dataset. Lines were shuffled within files. From step 81,000 a separate reader served the cooldown blend, which mixes datasets in every batch, and one batch in five carried a synthetic rewriting task built from the prose datasets. Over the run the model processed <span id="xTok"></span>. No dataset is read more than once.</p>
</section>"""
_a = body.index('<section id="data">'); _b = body.index('</section>', _a) + len('</section>')
body = body[:_a] + DATA_SECTION + body[_b:]

for _old, _new in REWRITES["replace"]:
    assert _old in body, f"rewrite rule does not match the report: {_old!r}"
    body = body.replace(_old, _new)

_m = re.search(r"const D = (\{.*?\});\n", body, re.S)
_D = json.loads(_m.group(1))
body = body.replace("label:c=>fmt(c.raw,2)+' B tokens'", "label:c=>(c.raw<0.01?fmt(c.raw*1000,2)+' M tokens':fmt(c.raw,2)+' B tokens')",1)
_D["pre"]["corpus"] = [[n, t] for n, _, t in BUDGET]
_D["q"]["sources"] = [x for x in _D["q"]["sources"] if x["name"] in ("All sources", "Educational web", "General web", "Code", "Textbooks", "Books")]
for _row in _D["hp"]:
    if _row[0] == "Cooldown data":
        _row[1] = "Educational web, textbooks, code and command-line references; 20% of batches a synthetic rewriting task"
body = body[:_m.start(1)] + json.dumps(_D, ensure_ascii=False, separators=(",", ":")) + body[_m.end(1):]

DECAY_SECTION = """  <h3>The decay stage: cooldown data and the synthetic rewriting task</h3>
  <p>The last 9,000 steps (81,000 to 90,000) changed two things at once: the learning rate decayed to zero, and the data stopped following the on-disk order. A cooldown reader drew documents from a curated blend in every batch, weighted toward educational web text, textbooks, permissively licensed code and command-line references. Each source was capped at about eight passes over itself so the small command-line sets were not memorised. The reason for changing the data as well as the learning rate is that the decay alone mostly lowers loss, while an up-weighted high-value mixture is what tends to move downstream quality.</p>
  <p>One batch in five (20%, drawn per batch with a seeded generator) was not a UL2 batch at all. It was a <b>synthetic rewriting exercise</b> in the exact shape the product uses: the encoder receives <code>&lt;style:grammar&gt; &lt;text_to_rewrite&gt; damaged text &lt;/text_to_rewrite&gt;</code> and the decoder must produce the original text. The pairs are built on the fly from ordinary prose documents from the pretraining datasets, so the model meets the rewriting frame, and learns to copy what is already correct, before any fine-tuning. The reserved style and frame tokens were already in the frozen vocabulary, so this needed no change to the embedding table.</p>
  <h4>How the text is damaged</h4>
  <p>The one rule is that the original must be the unique right answer. Every operator produces a genuine writing error whose correction is the source text. There is deliberately no synonym swapping: replacing "big" with "large" is not a mistake, and a target that reverses it would teach the model to edit text that was never wrong. Code, URLs, paths, numbers, identifiers, e-mail addresses and anything in backticks or angle brackets are protected spans that pass through unchanged on both sides, and lines that are not prose (code, tables, headings, bare links) are copied verbatim, so every example also trains copy-through.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Operator</th><th>Example of the damage</th><th>Weight</th></tr></thead>
    <tbody>
    <tr><td>Confusable word</td><td>their / there, is / are, went / gone, its / it's</td><td>4.0</td></tr>
    <tr><td>Dropped function word</td><td>"cat sat on mat"</td><td>3.0</td></tr>
    <tr><td>Typo</td><td>transposed, missing, doubled or keyboard-neighbour letter; first letter kept</td><td>3.0</td></tr>
    <tr><td>Capitalisation</td><td>"paris", "i think"</td><td>2.0</td></tr>
    <tr><td>Preposition swap</td><td>in / on / at, to / for</td><td>1.5</td></tr>
    <tr><td>Dropped apostrophe</td><td>"dont", "Johns"</td><td>1.5</td></tr>
    <tr><td>Doubled word</td><td>"the the"</td><td>1.0</td></tr>
    <tr><td>Swapped neighbours</td><td>"sat the cat"</td><td>1.0</td></tr>
    <tr><td>Run-on sentence (per document habit)</td><td>comma splice, dropped full stop, or missing capital</td><td>by profile</td></tr>
    </tbody></table></div>
  <p>Three things stop the inverse from being a lookup table. Severity is sampled per document from three profiles, so the model cannot assume how much is wrong. The operator is sampled per word, so the same word is a homophone swap in one document and a typo in the next. And a slice of documents is left untouched, because "this text is already correct, copy it" is a case the product meets constantly.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Profile</th><th>Probability</th><th>Errors per word</th><th>Lowercased</th><th>Punctuation stripped</th><th>Run-ons</th><th>Pairs seen</th></tr></thead>
    <tbody>
    <tr><td>Light</td><td>35%</td><td>3–8%</td><td>5%</td><td>0%</td><td>10%</td><td>430,220</td></tr>
    <tr><td>Medium</td><td>40%</td><td>8–18%</td><td>15%</td><td>10%</td><td>25%</td><td>501,451</td></tr>
    <tr><td>Heavy</td><td>25%</td><td>18–35%</td><td>40%</td><td>35%</td><td>40%</td><td>314,445</td></tr>
    <tr><td>Untouched (copy)</td><td>5% target</td><td>0</td><td>–</td><td>–</td><td>–</td><td>79,280</td></tr>
    </tbody></table></div>
  <p>Documents with fewer than eight prose words are skipped, and a document that does not fit the encoder or decoder length budget is skipped and counted rather than cut. In the run, the task read 1,376,936 documents, skipped 51,540 as not prose (3.7%), skipped none for length, and emitted <b>1,325,396 pairs</b> containing 80.1 million damaged words, about 60 per pair, from 0.55 billion source tokens. The untouched share came out at 6.0% against the 5% target. Its training loss fell to about 0.05 nats per token by the end (Figure 15), far below the 1.0 to 2.0 of the UL2 denoisers, because most of the output is a copy of the input. That is the intended behaviour: the task teaches precise edits inside a faithful copy, not new text.</p>
  <h4>Why this is in pretraining</h4>
  <p>The rewriting adapters start from a model that has already seen the style token and the text-to-rewrite frame, instead of meeting them cold in fine-tuning. The reader's random state and position are saved in every checkpoint, so an interrupted run replays the same damage on resume.</p>
"""
body = body.replace('  <h3>Optimizer parameter groups</h3>', DECAY_SECTION + '  <h3>Optimizer parameter groups</h3>', 1)

body = body.replace('<nav class="toc" aria-label="Contents">', '<nav class="toc" id="toc" aria-label="Contents">', 1)
updated = dt.date.today().strftime("%-d %B %Y")
body = re.sub(r"(<footer>.*?)(</footer>)", lambda m: m.group(1) + f'<div class="meta">Page generated {updated}.</div>' + m.group(2), body, count=1, flags=re.S)
tail = '<button class="btn top" id="top" type="button" aria-label="Back to top">Top ↑</button>\n<script src="assets/site.js"></script>\n'
# site.js must run after the report's own script; both sit at the end of the body
page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{TITLE}</title>
<meta name="description" content="{DESC}">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESC}">
<meta property="og:type" content="article">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="assets/fonts.css">
{style}
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
{bar}{body}
{tail}</body>
</html>
'''
# the report's own <script> blocks end with </script>; site.js is appended after them
_hits = sorted({m.group(0) for m in FORBIDDEN.finditer(page)})
if _hits:
    sys.exit(f"refusing to write index.html: forbidden text still present: {_hits}")
(HERE / "index.html").write_text(page)

(HERE / "404.html").write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Page not found</title><meta http-equiv="refresh" content="4; url=./">
<style>body{{font:16px/1.6 system-ui,sans-serif;margin:0;display:grid;place-items:center;min-height:100vh;background:#fbfbfa;color:#14161a}}
@media (prefers-color-scheme:dark){{body{{background:#0e1013;color:#e9ebee}}}}a{{color:#1f4fd8}}@media (prefers-color-scheme:dark){{a{{color:#7f9cff}}}}</style></head>
<body><p>That page does not exist. <a href="./">Go to the report</a>.</p></body></html>
''')
(HERE / "favicon.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="6" fill="#1f4fd8"/><path d="M7 21l5-6 4 3 5-8 4 5" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>\n')
(HERE / ".nojekyll").write_text("")
print("wrote index.html", len(page), "bytes")
