"""Pick the tests each change can break, from a map of which tests execute which lines of sqlglot."""

import base64
import gzip
import json
import os
import re
import sys
from bisect import bisect_right
from pathlib import Path

DATA = Path(os.environ.get("BR_DATA", "/data"))
MAP = Path(os.environ.get("BR_MAP", Path(__file__).with_name("map.json.gz")))
WORKER_MS = 2000
MAX_WORKERS = 8
# Files whose change can alter how pytest runs every test.
CONFIG_FILES = {"setup.cfg", "pyproject.toml", "setup.py", "conftest.py", "tests/conftest.py", "pytest.ini", "tox.ini"}


def parse_patch(text):
    """[(old_path, new_path, blocks)]. A block is one run of changed lines: (removed [(old_line, text)],
    added [text], at) where `at` is the old line the added lines go before."""
    files = []  # [old, new, blocks]; "" stands for /dev/null until the end
    cur = None
    old = 0
    old_left = new_left = 0
    block = None

    def close():
        nonlocal block
        if block is not None and cur is not None:
            cur[2].append(block)
        block = None

    def strip(path):
        path = path.split("\t")[0].strip()
        if path == "/dev/null":
            return ""
        return path[2:] if path[:2] in ("a/", "b/") else path

    for line in text.splitlines():
        if old_left > 0 or new_left > 0:
            # Inside a hunk the counts decide, whatever a line starts with ("--- x" can be a removed "-- x").
            if line.startswith("\\"):
                continue
            tag, body = (line[:1], line[1:]) if line else (" ", "")
            if tag == "-":
                if block is None:
                    block = ([], [], old)
                block[0].append((old, body))
                old += 1
                old_left -= 1
            elif tag == "+":
                if block is None:
                    block = ([], [], old)
                block[1].append(body)
                new_left -= 1
            else:
                close()
                old += 1
                old_left -= 1
                new_left -= 1
            continue
        close()
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            cur = [m.group(1), m.group(2), []] if m else [None, None, []]
            files.append(cur)
        elif line.startswith("--- ") and cur is not None:
            cur[0] = strip(line[4:])
        elif line.startswith("--- "):
            cur = [strip(line[4:]), None, []]
            files.append(cur)
        elif line.startswith("+++ ") and cur is not None:
            cur[1] = strip(line[4:])
        elif line.startswith("new file mode") and cur is not None:
            cur[0] = ""
        elif line.startswith("deleted file mode") and cur is not None:
            cur[1] = ""
        elif line.startswith("rename from ") and cur is not None:
            cur[0] = line[len("rename from "):].strip()
        elif line.startswith("rename to ") and cur is not None:
            cur[1] = line[len("rename to "):].strip()
        else:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m and cur is not None:
                old = int(m.group(1))
                old_left = int(m.group(2)) if m.group(2) is not None else 1
                new_left = int(m.group(4)) if m.group(4) is not None else 1
                if old_left == 0:
                    old += 1  # "-n,0" names the line the insertion follows
    close()
    return [(o or None, n or None, blocks) for o, n, blocks in files]


class Spans:
    """Sorted [start, end, ...] rows; find the row whose range holds a line."""

    def __init__(self, rows):
        self.rows = sorted(rows)
        self.starts = [r[0] for r in self.rows]

    def find(self, line):
        i = bisect_right(self.starts, line) - 1
        while i >= 0:
            r = self.rows[i]
            if r[0] <= line <= r[1]:
                return r
            if r[1] < line - 400:
                break
            i -= 1
        return None


class Map:
    def __init__(self, path):
        with gzip.open(path, "rb") as fh:
            self.d = json.loads(fh.read())
        self.n = self.d["n_tests"]
        self._sets = {}
        self._spans = {}

    def set(self, sid):
        s = self._sets.get(sid)
        if s is None:
            s = int.from_bytes(base64.b64decode(self.d["sets"][sid]), "little")
            self._sets[sid] = s
        return s

    def spans(self, path, kind):
        key = (path, kind)
        if key not in self._spans:
            self._spans[key] = Spans(self.d["files"][path][kind])
        return self._spans[key]

    def all_tests(self):
        return (1 << self.n) - 1


def is_comment_or_blank(text):
    s = text.strip()
    return not s or s.startswith("#")


def block_lines(block):
    """Old lines a block touches: the removed ones, or for an insertion the lines on either side."""
    removed, added, at = block
    if removed:
        return [ln for ln, _ in removed], False
    return [x for x in (at - 1, at) if x > 0], True


def select_sqlglot(m, path, blocks):
    info = m.d["files"][path]
    strings = m.spans(path, "str")
    rt, imp = m.spans(path, "rt"), m.spans(path, "imp")
    sel = 0
    for block in blocks:
        removed, added, at = block
        lines, insertion = block_lines(block)
        texts = [t for _, t in removed] + added
        if insertion:
            # Inserted before old line `at`: inside a string only if the string runs across that gap.
            s = strings.find(at)
            in_string = s is not None and s[0] < at
        else:
            in_string = any(strings.find(x) for x in lines)
        # Only comments or blank lines, none inside a multi-line string: nothing runs differently.
        if all(is_comment_or_blank(t) for t in texts) and not in_string:
            continue
        hits = [(x, rt.find(x), imp.find(x)) for x in lines]
        if insertion:
            runtime = [r for _, r, _ in hits if r is not None]
            # Between two statements of a function body (or after its last one): it runs where they run.
            if runtime and all(i is None for _, _, i in hits):
                for r in runtime:
                    sel |= m.set(r[2])
                continue
        for x, r, i in hits:
            if r is not None:
                sel |= m.set(r[2])
            if r is None or insertion:
                if i is not None and len(i) > 2 and not insertion:
                    sel |= m.set(i[2])  # a load-time statement, priced by who reads what it defines
                elif i is not None or not insertion:
                    sel |= m.set(info["importers"])
    return sel


def select_testfile(m, path, blocks):
    spans = Spans(m.d["testfiles"].get(path, []))
    sel = 0
    for block in blocks:
        removed, added, at = block
        texts = [t for _, t in removed] + added
        if all(is_comment_or_blank(t) for t in texts):
            continue
        lines, insertion = block_lines(block)
        rows = [spans.find(x) for x in lines]
        if rows and all(r is not None and r[2] for r in rows):
            for r in rows:
                for n in r[2]:
                    sel |= 1 << n
        else:
            if path in m.d["file_tests"]:
                sel |= m.set(m.d["file_tests"][path])
            if path in m.d["opens"]:
                sel |= m.set(m.d["opens"][path])
    return sel


def select(m, patch_text):
    sel = 0
    for old, new, blocks in parse_patch(patch_text):
        path = old or new
        if not path:
            continue
        if path in CONFIG_FILES:
            return m.all_tests()
        # Tests that read the file as data (source parsed as text, fixtures, setup scripts).
        for p in {old, new} - {None}:
            if p in m.d["opens"]:
                sel |= m.set(m.d["opens"][p])
        if path.startswith("sqlglot/") and path.endswith(".py"):
            if old is None:
                continue  # a new module runs only where a changed module imports it
            if path not in m.d["files"]:
                return m.all_tests()
            if new is None or (new and new != old):
                sel |= m.set(m.d["files"][path]["importers"])
                continue
            sel |= select_sqlglot(m, path, blocks)
        elif path.startswith("tests/") and path.endswith(".py") and path in m.d["testfiles"] and old is not None:
            sel |= select_testfile(m, path, blocks)
        elif path.startswith("tests/"):
            if old is not None and path in m.d["opens"]:
                sel |= m.set(m.d["opens"][path])
            if old is None or new is None or (new and new != old):
                # A file added, removed or renamed in a folder some test reads: those tests may list the folder.
                for p in {path, new} - {None}:
                    folder = p.rsplit("/", 1)[0] + "/"
                    for k, sid in m.d["opens"].items():
                        if k.startswith(folder):
                            sel |= m.set(sid)
        elif path.startswith("sqlglot/"):
            return m.all_tests()
    return sel


def pack(tests, ms):
    """The cheapest split: for 1..8 workers, longest test first onto the lightest worker."""
    if not tests:
        return []
    order = sorted(tests, key=lambda t: -ms[t])
    best = None
    for k in range(1, min(MAX_WORKERS, len(order)) + 1):
        loads = [0] * k
        workers = [[] for _ in range(k)]
        for t in order:
            i = loads.index(min(loads))
            loads[i] += ms[t]
            workers[i].append(t)
        cost = max(loads) + WORKER_MS * k
        if best is None or cost < best[0]:
            best = (cost, workers)
    return best[1]


def main():
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    ms = {t["number"]: round(t["seconds"] * 1000) for t in tests}
    m = Map(MAP)
    for patch in sorted((DATA / "commits").glob("*.patch")):
        try:
            mask = select(m, patch.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # never crash a run: run everything for this commit
            print(f"{patch.stem}: {exc!r}", file=sys.stderr)
            mask = m.all_tests()
        chosen = [n for n in ms if mask >> n & 1]
        print(json.dumps({"commit": patch.stem, "workers": pack(chosen, ms)}))


if __name__ == "__main__":
    main()
