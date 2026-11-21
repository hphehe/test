"""Pick the tests each change can break, from a map of which tests execute which lines of sqlglot."""

import ast
import base64
import gzip
import json
import os
import re
import sys
import textwrap
import warnings
import zlib
from bisect import bisect_right
from pathlib import Path

DATA = Path(os.environ.get("BR_DATA", "/data"))
MAP = Path(os.environ.get("BR_MAP", Path(__file__).with_name("map.json.gz")))
WORKER_MS = 2000
MAX_WORKERS = 8
# Probe: a commit whose selection would cost more than this runs nothing (measures what those commits cost us).
PROBE_MAX_MS = 30000
# Files whose change can alter how pytest runs every test.
CONFIG_FILES = {"setup.cfg", "pyproject.toml", "setup.py", "conftest.py", "tests/conftest.py", "pytest.ini", "tox.ini"}
# A line opening one of these clauses extends the statement before it rather than starting a new one.
CLAUSE = re.compile(r"(else|elif|except|finally|case)\b")
FRAGMENT_BINDS = re.compile(r"\w+\s*:=|\byield\b|\bawait\b")
# Comparison operators, in the order the map's per-site masks use.
CMP_OPS = ["is", "is not", "==", "!=", "<", "<=", ">", ">="]
CMP_RE = re.compile(r"\bis not\b|\bis\b|==|!=|<=|>=|<|>")


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


class Blocks:
    """Runtime blocks of one file, [lo, hi, col, fn set, [[start, end, set], ...]], looked up by column."""

    def __init__(self, rows):
        self.by_col = {}
        for r in rows:
            self.by_col.setdefault(r[2], []).append(r)
        self.los = {c: [r[0] for r in rs] for c, rs in self.by_col.items()}

    def find(self, col, at):
        """The block that code at indentation `col`, inserted before old line `at`, would join."""
        rows = self.by_col.get(col)
        if not rows:
            return None
        i = bisect_right(self.los[col], at) - 1
        while i >= 0:
            if rows[i][0] <= at <= rows[i][1]:
                return rows[i]
            i -= 1
        return None

    def innermost(self, at):
        """The deepest block whose lines run across old line `at`."""
        best = None
        for col in self.by_col:
            b = self.find(col, at)
            if b is not None and (best is None or col > best[2]):
                best = b
        return best


class Map:
    def __init__(self, path):
        with gzip.open(path, "rb") as fh:
            self.d = json.loads(fh.read())
        self.n = self.d["n_tests"]
        self._sets = {}
        self._spans = {}
        self._blocks = {}

    def blocks(self, path):
        if path not in self._blocks:
            self._blocks[path] = Blocks(json.loads(self.d["files"][path].get("blk", "[]")))
        return self._blocks[path]

    def same_results(self, path):
        """{"line:col": mask}: at the comparison whose operator starts there, bit j says CMP_OPS[j] gave the same
        result as the operator written, on every evaluation while the whole suite ran at the pinned commit."""
        key = (path, "cmp")
        if key not in self._blocks:
            self._blocks[key] = json.loads(self.d["files"][path].get("cmp", "{}"))
        return self._blocks[key]

    def swap_sets(self, path):
        """{"line:col": {"j": sid}}: at that comparison, the tests in which CMP_OPS[j] gave a different result than
        the operator written (or could not be checked) at least once; recorded test by test, each in its own process."""
        key = (path, "cmpt")
        if key not in self._blocks:
            self._blocks[key] = json.loads(self.d["files"][path].get("cmpt", "{}"))
        return self._blocks[key]

    def source(self, path):
        """The pinned commit's text of a sqlglot/ or tests/ Python file, or None."""
        b = self.d.get("src", {}).get(path)
        return zlib.decompress(base64.b64decode(b)).decode("utf-8") if b is not None else None

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


def indent(text):
    return len(text) - len(text.lstrip())


def bound_names(texts):
    """Names some code binds, plus markers for yield and await (which change the whole function), or None when the
    code does not parse on its own (a fragment of a longer statement)."""
    src = textwrap.dedent("\n".join(t for t in texts if not is_comment_or_blank(t)))
    last = src.splitlines()[-1] if src else ""
    if re.match(r"(else|elif)\b", src):
        src = "if _:\n    pass\n" + src  # a clause parses only after the statement it extends
    elif re.match(r"(except|finally)\b", src):
        src = "try:\n    pass\n" + src
    tree = None
    for attempt in (src, f"{src}\n{' ' * (indent(last) + 4)}pass"):  # the second closes a header line
        try:
            tree = ast.parse(attempt)
            break
        except Exception:
            continue
    if tree is None:
        return None
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update("global " + n for n in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        elif isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            names.add("<" + type(node).__name__ + ">")
    return names


def scoping_changed(removed, added, fragment):
    """Whether an edit may bind or unbind a name (a function's locals are decided by its whole body).
    `fragment`: the edit lies inside one statement."""
    old = [t for _, t in removed if not is_comment_or_blank(t)]
    new = [t for t in added if not is_comment_or_blank(t)]
    if not fragment:
        a = bound_names(new) if new else set()
        r = bound_names(old) if old else set()
        if a is not None and r is not None:
            return a != r
        if not old:
            return True  # inserted code that does not parse on its own may bind anything
    # Inside a statement only `:=` binds; yield and await change the whole function.
    return sorted(FRAGMENT_BINDS.findall("\n".join(old))) != sorted(FRAGMENT_BINDS.findall("\n".join(new)))


def operator_swap(old, new):
    """(column, index in CMP_OPS) when `new` is `old` with one comparison operator replaced by another, else None."""
    for mt in CMP_RE.finditer(old):
        for j, op in enumerate(CMP_OPS):
            if op != mt.group() and old[: mt.start()] + op + old[mt.end():] == new:
                return mt.start(), j
    return None


def same_everywhere(m, path, removed, added):
    """True when every changed line only swaps a comparison operator for one that gave the same result at that
    comparison on every evaluation the test suite makes: then no test can behave differently."""
    if not removed or len(removed) != len(added):
        return False
    sites = m.same_results(path)
    swapped = False
    for (x, old), new in zip(removed, added):
        if old == new:
            continue
        sw = operator_swap(old, new)
        if sw is None or not sites.get(f"{x}:{sw[0]}", 0) >> sw[1] & 1:
            return False
        swapped = True
    return swapped


def swap_tests(m, path, removed, added):
    """When every changed line only swaps one comparison operator for another: the tests in which the new operator
    gave a different result at that comparison at least once (0 if it agreed everywhere). A test outside them
    evaluates every changed comparison the same way, so it runs exactly as before. None when some line is not such a
    swap or the swap has no per-test record (the caller prices the block as before)."""
    if not removed or len(removed) != len(added):
        return None
    agree, per_test = m.same_results(path), m.swap_sets(path)
    sel, swapped = 0, False
    for (x, old), new in zip(removed, added):
        if old == new:
            continue
        sw = operator_swap(old, new)
        if sw is None:
            return None
        swapped = True
        key = f"{x}:{sw[0]}"
        if agree.get(key, 0) >> sw[1] & 1:
            continue
        sid = per_test.get(key, {}).get(str(sw[1]))
        if sid is None:
            return None
        sel |= m.set(sid)
    return sel if swapped else None


def land(m, path, at, col, clause):
    """Tests that can reach code put before old line `at` at indentation `col`, or None when that code would run at
    load time (module or class level) or cannot be placed. `clause`: the code opens an else/except/... clause."""
    b = m.blocks(path).find(col, at)
    if b is None:
        return None
    prev = nxt = None
    for s in b[4]:
        if s[0] < at:
            prev = s
        else:
            nxt = s
            break
    if prev is not None and (clause or nxt is None or prev[1] >= at):
        return m.set(prev[2])  # it can run only after `prev` started
    if nxt is not None and not clause:
        return m.set(nxt[2])  # it runs exactly when `nxt` would have started
    return None


def header_change(m, path, removed, added):
    """Line-for-line edits inside function headers that run at load. An edit inside an annotation costs nothing
    (the map lists annotation spans only for modules with `from __future__ import annotations`, where they stay
    strings nothing evaluates); an edit inside a literal default that leaves a literal costs the function's
    callers. None for anything else (the caller prices it as before)."""
    rows = m.d["files"][path].get("hdr")
    if not rows or not removed or len(removed) != len(added):
        return None
    sel = 0
    for (x, old), new in zip(removed, added):
        if old == new:
            continue
        row = next((r for r in rows if r[0] <= x <= r[1]), None)
        if row is None:
            return None
        n = min(len(old), len(new))
        a = 0
        while a < n and old[a] == new[a]:
            a += 1
        q = 0
        while q < n - a and old[-1 - q] == new[-1 - q]:
            q += 1
        b = len(old) - q  # old[a:b] became new[a:len(new) - q]

        def inside(spans):
            for l0, c0, l1, c1 in spans:
                lo, hi = (c0 if x == l0 else 0), (c1 if x == l1 else len(old))
                if l0 <= x <= l1 and lo <= a and b <= hi:
                    return l0, lo, l1, hi
            return None

        if inside(row[3]) is not None:
            continue
        got = inside(row[4])
        if got is None or not got[0] == got[2] == x:
            return None  # not in a default, or a default over several lines
        lo, hi = got[1], got[3]
        try:
            ast.literal_eval(new[lo:hi + len(new) - len(old)].strip())
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return None  # no longer a literal: evaluating it at load may fail
        sel |= m.set(row[2])
    return sel


def select_sqlglot(m, path, blocks):
    info = m.d["files"][path]
    strings = m.spans(path, "str")
    rt, imp = m.spans(path, "rt"), m.spans(path, "imp")
    importers = m.set(info["importers"])

    def statement(r, i):
        """A change to a statement: who runs it, or who loads the module if it runs at load time."""
        if r is not None:
            return m.set(r[2])
        if len(i) > 2:
            return m.set(i[2])  # a load-time statement, priced by who reads what it defines
        return importers

    sel = 0
    for removed, added, at in blocks:
        texts = [t for _, t in removed] + added
        if removed:
            in_string = any(strings.find(x) for x, _ in removed)
        else:
            s = strings.find(at)  # inserted before old line `at`: inside a string only if it runs across that gap
            in_string = s is not None and s[0] < at
        # Only comments or blank lines, none inside a multi-line string: nothing runs differently.
        if all(is_comment_or_blank(t) for t in texts) and not in_string:
            continue
        got = swap_tests(m, path, removed, added)
        if got is not None:
            sel |= got  # 0 when every swap agreed in every test
            continue
        got = header_change(m, path, removed, added)
        if got is not None:
            sel |= got  # 0 when only annotations changed
            continue
        # Removed lines: each changes or deletes the statement it belongs to.
        for x, t in removed:
            if is_comment_or_blank(t) and not strings.find(x):
                continue
            r, i = rt.find(x), imp.find(x)
            if r is not None or i is not None:
                sel |= statement(r, i)
            else:
                got = land(m, path, x, indent(t), True)  # a clause line (`else:`): price the statement it extends
                sel |= importers if got is None else got
        # Added lines: inside a statement that runs across the gap they belong to it; else place them by indentation.
        code = [t for t in added if not is_comment_or_blank(t)]
        r, i = rt.find(at), imp.find(at)
        r = r if r is not None and r[0] < at else None
        i = i if i is not None and i[0] < at else None
        if r is not None or i is not None:
            if not removed:
                sel |= statement(r, i)
        elif code:
            first_old = next((t for _, t in removed if not is_comment_or_blank(t)), None)
            in_place = indent(first_old) if first_old is not None else None
            low = None
            for t in code:
                c = indent(t)
                if low is not None and c >= low:
                    continue  # nested in (or continuing) an added line above
                low = c
                if c == in_place:
                    continue  # takes the removed statements' place: priced above
                got = land(m, path, at, c, bool(CLAUSE.match(t.lstrip())))
                sel |= importers if got is None else got
        # A new or dropped name binding changes scoping in the whole function (yield makes it a generator).
        if not (in_string and not removed) and scoping_changed(removed, added, r is not None or i is not None):
            first = next((t for t in [t for _, t in removed] + added if not is_comment_or_blank(t)), "")
            b = m.blocks(path).find(indent(first), at)
            if b is None and r is not None:
                b = m.blocks(path).innermost(at)
            sel |= importers if b is None else m.set(b[3])
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


def inert_module(blocks):
    """True when a new module's code, run at import, only binds names: imports, defs, classes whose bodies only do
    the same, plain assignments, docstrings and a __main__ guard. Loading such a test module cannot change how
    other tests run, and its own tests are not in the list."""
    try:
        tree = ast.parse("\n".join(t for b in blocks for t in b[1]))
    except (SyntaxError, ValueError):
        return False

    def plain_value(v):
        if isinstance(v, (ast.Name, ast.Attribute)):
            return True
        try:
            ast.literal_eval(v)
            return True
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return False

    def plain(s, in_class):
        if isinstance(s, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.Pass)):
            return True
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            return True
        if isinstance(s, ast.Assign) or (isinstance(s, ast.AnnAssign) and s.value is not None):
            return plain_value(s.value)
        if isinstance(s, ast.ClassDef):
            return all(isinstance(b, (ast.Name, ast.Attribute)) for b in s.bases) and \
                all(plain(x, True) for x in s.body)
        if not in_class and isinstance(s, ast.If) and isinstance(s.test, ast.Compare) and \
                isinstance(s.test.left, ast.Name) and s.test.left.id == "__name__":
            return True
        return False

    return all(plain(s, False) for s in tree.body)


def folder_readers(m, path):
    """Tests that read some file in the folder of `path`: they may list the folder (test_typing imports every
    module in sqlglot/typing; fixture loaders read whole folders), so a file added or removed there reaches them."""
    folder = path.rsplit("/", 1)[0] + "/"
    sel = 0
    for k, sid in m.d["opens"].items():
        if k.startswith(folder):
            sel |= m.set(sid)
    return sel


def patched(text, blocks):
    """`text` with the blocks applied, or None when a removed line does not match (the patch is for other text)."""
    lines = text.split("\n")
    out, i = [], 1  # i: the next old line to copy
    for removed, added, at in blocks:
        start = removed[0][0] if removed else at
        if start < i:
            return None
        out += lines[i - 1:start - 1]
        for ln, t in removed:
            if ln > len(lines) or lines[ln - 1].rstrip("\r") != t.rstrip("\r"):
                return None
        out += added
        i = removed[-1][0] + 1 if removed else start
    return "\n".join(out + lines[i - 1:])


def breaks_syntax(m, old, new, blocks):
    """True when the changed Python file no longer compiles. Python is the run's version (the image pins 3.13)."""
    if old is None:
        text = "\n".join(t for b in blocks for t in b[1]) + "\n"  # a new file is all added lines
    else:
        src = m.source(old)
        text = patched(src, blocks) if src is not None else None
        if text is None:
            return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            compile(text, new or old, "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            return True
        except Exception:  # noqa: BLE001  (recursion and memory limits of a huge expression: not our concern)
            return False
    return False


def select(m, patch_text):
    sel = 0
    for old, new, blocks in parse_patch(patch_text):
        path = old or new
        if path and new and path.endswith(".py") and (new or "").endswith(".py") and \
                path.startswith(("sqlglot/", "tests/")) and breaks_syntax(m, old, new, blocks):
            # A module that no longer compiles fails every test that imports it, and a test file that does not
            # compile stops collection (pytest's default interrupts the whole run): run everything.
            return m.all_tests()
        path = old or new
        if not path:
            continue
        paths = {old, new} - {None, ""}
        # Config, and any .py file at the root: the root is on sys.path in the test run, so a new json.py there
        # would shadow the standard library for every test.
        if any(p in CONFIG_FILES or ("/" not in p and p.endswith(".py")) for p in paths):
            return m.all_tests()
        if any(p.rsplit("/", 1)[-1] == "conftest.py" for p in paths):
            # A conftest below the root applies to every test under its folder.
            for p in paths:
                folder = p.rsplit("/", 1)[0] + "/"
                for f, sid in m.d["file_tests"].items():
                    if f.startswith(folder):
                        sel |= m.set(sid)
            continue
        # Tests that read the file as data (source parsed as text, fixtures, setup scripts).
        for p in paths:
            if p in m.d["opens"]:
                sel |= m.set(m.d["opens"][p])
        if path.startswith("sqlglot/") and path.endswith(".py"):
            if old is None or new is None or new != old:
                for p in paths:
                    sel |= folder_readers(m, p)
            if old is None:
                # A new module runs only where a changed module imports it, or a test lists its folder, or it is
                # a package that now shadows the module of the same name.
                if path.endswith("/__init__.py") and path[: -len("/__init__.py")] + ".py" in m.d["files"]:
                    sel |= m.set(m.d["files"][path[: -len("/__init__.py")] + ".py"]["importers"])
                continue
            if path not in m.d["files"]:
                return m.all_tests()
            if new is None or (new and new != old):
                sel |= m.set(m.d["files"][path]["importers"])
                continue
            sel |= select_sqlglot(m, path, blocks)
        elif path.startswith("tests/") and path.endswith(".py") and path in m.d["testfiles"] and old is not None:
            sel |= select_testfile(m, path, blocks)
        elif path.startswith("tests/"):
            if old is None and path.endswith(".py") and not path.endswith("/__init__.py") and inert_module(blocks):
                continue  # a new test module that only defines things (its tests are not in the list)
            if old is not None and path in m.d["opens"]:
                sel |= m.set(m.d["opens"][path])
            if old is None or new is None or (new and new != old):
                # A file added, removed or renamed in a folder some test reads: those tests may list the folder.
                for p in paths:
                    sel |= folder_readers(m, p)
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
        workers = pack(chosen, ms)
        if workers and max(sum(ms[t] for t in w) for w in workers) + WORKER_MS * len(workers) > PROBE_MAX_MS:
            workers = []
        print(json.dumps({"commit": patch.stem, "workers": workers}))


if __name__ == "__main__":
    main()
