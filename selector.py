"""Pick the tests each change can break, from a map of which tests execute which lines of sqlglot."""

import ast
import base64
import copy
import gzip
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from bisect import bisect_right
from pathlib import Path

DATA = Path(os.environ.get("BR_DATA", "/data"))
MAP = Path(os.environ.get("BR_MAP", Path(__file__).with_name("map.json.gz")))
WORKER_MS = 2000
MAX_WORKERS = 8
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
    # In-place edits of the TPC-H / TPC-DS query files: only the TestExecutor tests whose inputs changed (see
    # tpc_fixture_model). Without a model every reader of the file is kept, as before.
    try:
        fx = tpc_fixture_model(m, patch_text) if any(p in patch_text for p in TPC_FIX) else None
    except Exception:  # noqa: BLE001  (no model: every reader of the file is kept)
        fx = None
    fx_mask = -1
    if fx is not None:
        tx = m.d["texec"]
        fx_mask = ~sum(1 << int(k) for k in tx["ids"]) | fx["keep"] | ((1 << tx["tpcds"]) if fx["px"] else 0)

    def opens(p, in_place):
        s = m.set(m.d["opens"][p])
        return s & fx_mask if in_place and p in TPC_FIX else s

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
                sel |= opens(p, old == new)
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
                sel |= opens(path, old == new)
            if old is None or new is None or (new and new != old):
                # A file added, removed or renamed in a folder some test reads: those tests may list the folder.
                for p in paths:
                    sel |= folder_readers(m, p)
        elif path.startswith("sqlglot/"):
            return m.all_tests()
    return sel


# ---------------------------------------------------------------- TPC-DS check
# tests/test_executor.py::TestExecutor::test_execute_tpcds costs 21.5 s and runs most of the optimizer, so it is
# selected for many changes, yet few of them change its outcome. The test parses every TPC-DS query (parent process),
# executes those marked `execute` (child process: execute() = ensure_schema, optimize, a debug
# expression.sql(pretty=True), Plan, the Python executor) and compares each result with duckdb running the query as
# sqlglot transpiles it.
# For a change that only edits function bodies, the check reruns the cheap stages on the patched code: parse (and
# pickle) the affected queries, optimize them exactly as execute() does, render the debug sql, transpile to duckdb.
# The map holds, per sqlglot line, which queries ran it in which stage, and the outputs at the pinned commit. When
# every output is equal and no changed function runs after optimize (Plan, executor, tables) nor at import or in the
# test's table setup, the executor gets the same input and runs the same code, so the outcome cannot change: the
# test is dropped. Any doubt (other files, a line the stage record never saw, an error, a timeout) keeps it.

TPCDS_ID = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
TPCDS_DEPS = ("tests/test_executor.py", "tests/helpers.py", "tests/__init__.py", "tests/fixtures/optimizer/tpc-ds/",
              "tests/fixtures/optimizer/tpc-h/")
SQ = os.environ.get("BR_SQ", "/app/sq")  # the pinned sqlglot tree with .pyc, written by --prepare at build time
TP = os.environ.get("BR_TP", "/app/tp")  # the TPC-DS tables as the test builds them, written by --prepare
CHECK_JOBS = 2
CHECK_BUDGET_S = float(os.environ.get("BR_CHECK_BUDGET", 100))  # no check starts after this many seconds of the run
CHECK_MAX_EST_S = 14  # nor one estimated to cost more CPU seconds than this


# ---------------------------------------------------------------- TPC query files
# tests/fixtures/optimizer/tpc-h/tpc-h.sql and tpc-ds/tpc-ds.sql are read by TestExecutor.setUpClass, so every test of
# that class (60 tests, 34 s) reads them, and by test_optimizer's test_tpch / test_tpcds. Inside the class only three
# tests use what was loaded: test_execute_tpch and test_execute_tpcds (each query's meta and sql; queries without
# `execute` are only parsed and pickled), test_optimized_tpch (tpc-h sql and expected sql). The three share a result
# cache keyed by the sql text alone. Every other test of the class sees the files only through setUpClass succeeding.
# So for an in-place edit, replay the loader (tests/helpers.py) on the pinned and the edited text: if it raises,
# narrow nothing; otherwise keep only the tests whose inputs changed (test_optimizer's readers are always kept).
TPC_FIX = ("tests/fixtures/optimizer/tpc-h/tpc-h.sql", "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql")


def fixture_pairs(text):
    """tests/helpers.py load_sql_fixture_pairs with _filter_comments and _extract_meta, as a list, on the text that
    open(path, encoding="utf-8").read() returns. It raises where they raise."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # universal newlines, as a read in text mode does
    statements = "\n".join([line for line in text.splitlines() if line and not line.startswith("--")]).split(";")
    size = len(statements)
    out = []
    for i in range(0, size, 2):
        if i + 1 < size:
            sql_lines = statements[i].strip().split("\n")
            meta, j = {}, 0
            while sql_lines[j].startswith("#"):
                key, val = sql_lines[j].split(":", maxsplit=1)
                meta[key.lstrip("#").strip()] = val.strip()
                j += 1
            out.append((meta, "\n".join(sql_lines[j:]), statements[i + 1].strip()))
    return out


def tpc_fixture_model(m, patch_text):
    """For a change that edits tpc-h.sql or tpc-ds.sql in place: {"keep": mask of the TestExecutor tests that can
    see the edit, "px": [{i, sql, old}] TPC-DS queries the test only parses and pickles, whose text changed}.
    None when it cannot narrow: no data in the map, a file added, removed or renamed, a patch that does not apply,
    undecodable bytes, a loader that raises, or two queries with the same text across the files (shared cache)."""
    tx = m.d.get("texec")
    if not tx:
        return None
    new = {}
    for old, new_path, blocks in parse_patch(patch_text):
        if not ({old, new_path} & set(TPC_FIX)):
            continue
        if old != new_path or old in new:
            return None
        src = m.source(old)
        text = patched(src, blocks) if src is not None else None
        if text is None or "�" in text:
            return None
        new[old] = text
    if not new:
        return None
    try:
        base = {p: fixture_pairs(m.source(p)) for p in TPC_FIX}
        cur = {p: fixture_pairs(new.get(p, m.source(p))) for p in TPC_FIX}
    except Exception:  # noqa: BLE001  (setUpClass would fail: every test of the class errors)
        return None
    h, ds = TPC_FIX
    if {s for _, s, _ in cur[h]} & {s for _, s, _ in cur[ds]}:
        return None
    keep, px = 0, []
    if [(mt, s) for mt, s, _ in cur[h]] != [(mt, s) for mt, s, _ in base[h]]:
        keep |= 1 << tx["tpch"]
    if [(s, e) for _, s, e in cur[h]] != [(s, e) for _, s, e in base[h]]:
        keep |= 1 << tx["opt_tpch"]
    b, c = base[ds], cur[ds]
    if len(b) != len(c):
        keep |= 1 << tx["tpcds"]
    else:
        for i, ((bm, bs, _), (cm, cs, _)) in enumerate(zip(b, c)):
            if (bm, bs) == (cm, cs):
                continue
            if bm.get("execute") or cm.get("execute"):
                keep |= 1 << tx["tpcds"]
                break
            px.append({"i": i, "sql": cs, "old": bs})
    if keep >> tx["tpcds"] & 1:
        px = []
    return {"keep": keep, "px": px}


def fingerprint(e):
    """Everything later stages can read from a sqlglot tree: class, args in order, links, type, meta, comments.
    Reads the slots directly, never sqlglot methods. Same function as mapbuild/tpcds_prof.py."""
    from sqlglot.expressions import Expr
    out = []
    g = object.__getattribute__

    def slot(x, name):
        try:
            return g(x, name)
        except AttributeError:
            return "<unset>"

    def walk(x, parent, key, idx):
        if isinstance(x, Expr):
            out.append(f"<{type(x).__module__}.{type(x).__qualname__}|{int(slot(x, 'parent') is parent)}|"
                       f"{slot(x, 'arg_key')!r}|{slot(x, 'index')!r}|")
            args = slot(x, "args")
            if not isinstance(args, dict):
                out.append(f"ARGS?{type(args).__name__}")
                args = {}
            for k, v in args.items():
                out.append(f"{k}=")
                if isinstance(v, list):
                    out.append("[")
                    for i, y in enumerate(v):
                        walk(y, x, k, i)
                        out.append(",")
                    out.append("]")
                else:
                    walk(v, x, k, None)
                out.append(";")
            t = slot(x, "_type")
            if isinstance(t, Expr):
                out.append("T:")
                walk(t, None, None, None)
            else:
                out.append(f"T:{t!r}")
            extra = getattr(x, "__dict__", None) or {}
            out.append(f"M:{slot(x, '_meta')!r}C:{slot(x, 'comments')!r}D:{sorted(extra.items())!r}>")
        elif isinstance(x, list):
            out.append("[" + ",".join(repr(y) for y in x) + "]")
        else:
            out.append(f"{type(x).__name__}:{x!r}")

    walk(e, None, None, None)
    return hashlib.sha1("".join(out).encode("utf-8", "surrogatepass")).hexdigest()


def outer_functions(tree):
    """[(qualname, node)] of the functions not nested in another function, in source order."""
    out = []

    def walk(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((prefix + n.name, n))
            elif isinstance(n, ast.ClassDef):
                walk(n.body, prefix + n.name + ".")
            else:
                for f in ("body", "orelse", "finalbody", "handlers", "cases"):
                    sub = getattr(n, f, None)
                    if isinstance(sub, list):
                        walk(sub, prefix)

    walk(tree.body, "")
    return out


def plain_value(n):
    """An expression whose evaluation at def time has no side effect: a constant, a name, an attribute of a name, a
    sign or `not` on one of those, or a tuple of them."""
    if isinstance(n, (ast.Constant, ast.Name)):
        return True
    if isinstance(n, ast.Attribute):
        return plain_value(n.value)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd, ast.Not)):
        return plain_value(n.operand)
    if isinstance(n, ast.Tuple):
        return all(plain_value(e) for e in n.elts)
    return False


def signature_change(a, b, lazy_annotations):
    """None when two defs have the same signature; True when only default values (new ones plain) or, in a module
    whose annotations are never evaluated (PEP 563), annotations differ; False otherwise. Parameter names and kinds
    must stay: sqlglot reads them (inspect in optimizer.py and executor/env.py), never defaults or annotations."""
    if ast.dump(a.args) == ast.dump(b.args) and ast.dump(a.returns or ast.Pass()) == ast.dump(b.returns or ast.Pass()):
        return None
    x, y = a.args, b.args
    groups = ("posonlyargs", "args", "kwonlyargs")
    names = lambda s: [[p.arg for p in getattr(s, g)] for g in groups] + \
        [s.vararg and s.vararg.arg, s.kwarg and s.kwarg.arg]  # noqa: E731
    if names(x) != names(y) or len(x.defaults) != len(y.defaults) or \
            [d is None for d in x.kw_defaults] != [d is None for d in y.kw_defaults]:
        return False
    for d0, d1 in zip(x.defaults + x.kw_defaults, y.defaults + y.kw_defaults):
        if d0 is not None and ast.dump(d0) != ast.dump(d1) and not plain_value(d1):
            return False
    ann = lambda s, r: [ast.dump(p.annotation) if p.annotation else "" for g in groups for p in getattr(s, g)] + \
        [ast.dump(v.annotation) if v and v.annotation else "" for v in (s.vararg, s.kwarg)] + \
        [ast.dump(r) if r else ""]  # noqa: E731
    if ann(x, a.returns) != ann(y, b.returns) and not lazy_annotations:
        return False
    return True


def changed_bodies(old_text, new_text):
    """Old-numbered line spans of the bodies of the functions a change edits, or None when anything outside
    function bodies differs (defs, decorators, classes, module or class level statements). A def whose only change
    is in default values or (lazy) annotations counts as an edit of its body: the body is what then behaves
    differently, wherever it runs."""
    try:
        old_t, new_t = ast.parse(old_text), ast.parse(new_text)
    except (SyntaxError, ValueError):
        return None
    of, nf = outer_functions(old_t), outer_functions(new_t)
    if [q for q, _ in of] != [q for q, _ in nf]:
        return None
    lazy = any(isinstance(n, ast.ImportFrom) and n.module == "__future__" and
               any(a.name == "annotations" for a in n.names) for n in old_t.body + new_t.body)
    spans = []
    for (_, a), (_, b) in zip(of, nf):
        sig = signature_change(a, b, lazy)
        if sig is False:
            return None
        if sig or ast.dump(ast.Module(body=a.body, type_ignores=[])) != \
                ast.dump(ast.Module(body=b.body, type_ignores=[])):
            spans.append((a.body[0].lineno, a.end_lineno))
    for t in (old_t, new_t):
        for _, n in outer_functions(t):
            n.body = [ast.Pass()]
            n.args = ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None,
                                   defaults=[])
            n.returns = None
    if ast.dump(old_t) != ast.dump(new_t):
        return None
    return spans


def line_span(m, path, ln):
    """(start, end, mask) of the statement span holding line ln in the map's runtime rows, or None."""
    for s, e, sid in m.d["files"][path]["rt"]:
        if s <= ln <= e:
            return s, e, m.set(sid)
    return None


def tpcds_job(m, patch_text, tp):
    """The check that can drop test_execute_tpcds for this change, or None when it cannot apply."""
    tq = m.d.get("tpq")
    if not tq:
        return None
    files, spans, px = {}, [], []
    fx = None
    for old, new, blocks in parse_patch(patch_text):
        path = old or new
        if not path:
            return None
        if path in TPC_FIX and old == new:
            # The query file: the model says whether the test's inputs changed; queries it only parses go in px.
            fx = fx or tpc_fixture_model(m, patch_text)
            if fx is None or fx["keep"] >> tp & 1 or m.d["texec"]["tpcds"] != tp:
                return None
            px = fx["px"]
            continue
        if path.startswith("sqlglot/"):
            if not (old and new == old and path.endswith(".py") and path in m.d["files"]):
                return None
            text = m.source(path)
            new_text = patched(text, blocks) if text is not None else None
            sp = changed_bodies(text, new_text) if new_text is not None else None
            if sp is None:
                return None
            files[path] = new_text
            spans += [(path, a, b) for a, b in sp]
            continue
        paths = {old, new} - {None, ""}
        for p in paths:
            if p.endswith(".py") or p.startswith(TPCDS_DEPS) or p in CONFIG_FILES or "/" not in p:
                return None
            if p in m.d["opens"] and m.set(m.d["opens"][p]) >> tp & 1:
                return None  # the test reads this file
            if old != new or old is None:
                return None  # a file added, removed or renamed outside sqlglot: the test may list a folder
    if not files and not px:
        return None
    lines = tq["lines"]
    p_m = o_m = g_m = d_m = e_m = 0
    for path, a, b in spans:
        fl = lines.get(path, {})
        for ln in range(a, b + 1):
            row = fl.get(str(ln))
            if row is None:
                sp = line_span(m, path, ln)
                if sp and sp[2] >> tp & 1 and not any(str(x) in fl for x in range(sp[0], sp[1] + 1)):
                    # The test runs this statement, but no line of it is in any recorded stage. (Line events
                    # fire on a statement's first line, so its other lines are often missing: that is fine.)
                    return None
                continue
            p, o, g, d, x, e = row
            if x:
                return None  # runs at import or in table setup
            p_m |= p
            o_m |= o
            g_m |= g
            d_m |= d
            e_m |= e  # runs in the execute stage of these queries: they are executed and their results compared
    ex = tq["exec_mask"]
    q1 = (p_m & ex) | o_m | g_m
    pn = p_m & ~ex
    if (q1 | d_m) & tq["unstable_mask"] or not (q1 | pn | d_m | e_m | len(px)):
        return None
    if e_m and not tables_ready():
        return None
    qs = tq["q"]
    est = 1.0 + 0.05 * len(px) + sum(qs[i]["cpu"] for i in range(len(qs)) if (q1 | d_m) >> i & 1) + \
        sum(qs[i]["xcpu"] for i in range(len(qs)) if e_m >> i & 1)
    if est > CHECK_MAX_EST_S:
        return None
    pick = lambda mask, keys: [{"i": i, **{k: qs[i][k] for k in keys}} for i in range(len(qs)) if mask >> i & 1]  # noqa: E731
    return {"files": files, "schema": tq["schema"], "est": est, "q1": pick(q1, ("sql", "fo", "res", "xcpu")),
            "pn": pick(pn, ("sql",)), "d": pick(d_m, ("sql", "duck")), "px": px,
            "xq": pick(e_m, ("sql", "res", "xcpu")),
            # CPU left for executing queries whose optimized tree turns out different (only known in the check)
            "xbudget": round(CHECK_MAX_EST_S - est, 3) if tables_ready() else 0.0}


def tables_ready():
    """The TPC-DS tables --prepare built (and checked against the hashes of the test's own setUpClass tables)."""
    return os.path.isfile(os.path.join(TP, "ok"))


def result_hash(table):
    """The executed query's result as the test compares it: column names, then every row's values and types."""
    return hashlib.sha1(repr((list(table.columns), [tuple(r) for r in table.rows])).encode("utf-8", "surrogatepass")
                        ).hexdigest()


def tree_depth(e):
    """Depth of a sqlglot tree through its args, read from the slots."""
    from sqlglot.expressions import Expr
    best, stack = 0, [(e, 1)]
    while stack:
        x, d = stack.pop()
        best = max(best, d)
        for v in object.__getattribute__(x, "args").values():
            for c in (v if isinstance(v, list) else [v]):
                if isinstance(c, Expr):
                    stack.append((c, d + 1))
    return best


def phase_limit(seconds):
    """Raise in this process after `seconds` of wall time (replaces any earlier limit)."""
    import signal

    def stop(*_):
        raise TimeoutError("check phase over its time limit")

    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def patched_tree(files):
    """A temporary copy of the pinned tree with these files replaced. The caller removes it."""
    tmp = tempfile.mkdtemp(prefix="tpc")
    shutil.copytree(os.path.join(SQ, "sqlglot"), os.path.join(tmp, "sqlglot"))  # copy2: the .pyc stay valid
    for path, text in files.items():
        with open(os.path.join(tmp, path), "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    return tmp


def tpcds_check():
    """Child process: stdin is a job from tpcds_job. Prints "same" last when every output equals the pinned one."""
    job = json.loads(sys.stdin.read())
    sys.dont_write_bytecode = True
    tmp = patched_tree(job["files"])
    try:
        return query_check(job, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def query_check(job, tmp):
    """The cheap stages of test_execute_tpcds on the tree at tmp (imported here), against the pinned outputs."""
    if True:
        sys.path.insert(0, tmp)
        import sqlglot
        if not os.path.abspath(sqlglot.__file__).startswith(tmp):
            print("wrong sqlglot", sqlglot.__file__)
            return 1
        # What the test's child imports (tests/test_executor.py), so the same modules are loaded.
        from sqlglot import exp, find_tables, parse_one, transpile  # noqa: F401
        from sqlglot.errors import ExecuteError  # noqa: F401
        from sqlglot.executor import env, execute  # noqa: F401
        from sqlglot.executor.python import Python, PythonExecutor  # noqa: F401
        from sqlglot.executor.table import Table, ensure_tables  # noqa: F401
        from sqlglot.optimizer import optimize  # noqa: F401
        from sqlglot.planner import Plan  # noqa: F401
        import sqlglot.executor as sx
        # A change can make the optimizer loop: stop each phase at a few times its estimate (CPU is scored).
        phase_limit(max(8.0, 3 * job["est"] + 2))
        for q in job["pn"]:
            pickle.loads(pickle.dumps(parse_one(q["sql"])))
        diffs = []
        static = {q["i"] for q in job.get("xq", [])}
        extra = 0.0
        for q in job["q1"]:
            e = pickle.loads(pickle.dumps(parse_one(q["sql"])))
            schema = sx.ensure_schema(copy.deepcopy(job["schema"]), dialect=None)
            o = sx.optimize(e, schema, leave_tables_isolated=True, dialect=None)
            if fingerprint(o) != q["fo"]:
                diffs.append(q)  # the executor gets another tree: execute the query and compare its result
                extra += 0.0 if q["i"] in static else q["xcpu"]
                if extra > job.get("xbudget", 0.0):  # it cannot be dropped any more: stop spending CPU
                    print("optimize differs", [x["i"] for x in diffs], f"executing them would cost {extra:.1f} s")
                    return 1
            o.sql(pretty=True)
        for q in job["d"]:
            if transpile(q["sql"], write="duckdb")[0] != q["duck"]:
                print("duckdb sql differs", q["i"])
                return 1
        for q in job.get("px", []):
            # A query the test only parses (parent) and pickles to a child that returns None for it: it passes when
            # both work. The tree may not be deeper than the pinned query's (pickling recurses).
            a = parse_one(q["sql"])
            pickle.loads(pickle.dumps(a))
            if tree_depth(a) > tree_depth(parse_one(q["old"])) + 3:
                print("parsed query deeper", q["i"])
                return 1
        run = {q["i"]: q for q in job.get("xq", []) + diffs}
        if run:
            # mp_execute in the test's child: the tables the query names, execute() with the shared schema, the
            # result pickled back to the parent, which compares it with duckdb's. Same result as at the pinned
            # commit, where the test passes -> same comparison.
            if not tables_ready():
                print("no tables")
                return 1
            phase_limit(max(5.0, 3 * sum(q["xcpu"] for q in run.values()) + 2))
            shared_schema = pickle.loads(pickle.dumps(job["schema"]))
            cache = {}
            for q in sorted(run.values(), key=lambda q: q["xcpu"]):
                expression = pickle.loads(pickle.dumps(parse_one(q["sql"])))
                tables = {}
                for t in find_tables(expression):
                    name = t.name
                    if name not in cache:
                        with open(os.path.join(TP, name + ".pkl"), "rb") as fh:
                            cache[name] = pickle.load(fh)
                    tables[name] = cache[name]
                table = pickle.loads(pickle.dumps(execute(expression, schema=shared_schema, tables=tables)))
                if result_hash(table) != q["res"]:
                    print("result differs", q["i"])
                    return 1
        print("same")
        return 0


# ---------------------------------------------------------------- class-level TPC-DS check
# A change outside function bodies (class-level assignments) runs while sqlglot imports. The check imports every
# sqlglot module from the patched tree and from the pinned one (at build time) and fingerprints every module global
# and class dict entry: functions by bytecode, constants, defaults, closures; containers by content and order; enum
# members by name and value. When the only differences are plain values of class attributes, the change can reach
# test_execute_tpcds only where it reads those attributes at run time: none for an attribute of a class the test
# never reads it from (the map's class price), else the queries and stages whose lines read the name, which the
# TPC-DS check then reruns. Anything else (a function, a module global, class structure, an import error, a deep
# or unreadable value, a read while the tables are built) keeps the test.

SD = os.environ.get("BR_SD", "/app/sd.json")  # import-time state of the pinned tree, written by --prepare
SNAP_EST_S = 1.5
PROBE_EST_S = 0.6  # CPU of one fresh-process import probe (python + import sqlglot + the statement)  # CPU of the patched-tree snapshot (import everything + fingerprints), upper estimate
SNAP_SKIP_CLASS = {"__dict__", "__weakref__", "__module__", "__qualname__", "__firstlineno__", "__static_attributes__",
                   "__doc__"}
SNAP_SKIP_MOD = {"__file__", "__cached__", "__spec__", "__loader__", "__builtins__", "__path__", "__name__",
                 "__package__", "__doc__"}


def snap_state(root, bag_keys=()):
    """Child process: import every sqlglot module from root with unittest loaded (sqlglot's UNITTEST flag, as under
    pytest) and fingerprint the state. ({key: [hash, tag, module, qualname]}, {key: bag hash}, stats)."""
    import enum
    import functools
    import gc
    import importlib
    import pkgutil
    import types
    import unittest  # noqa: F401
    sys.setrecursionlimit(20000)
    sys.path.insert(0, root)
    import sqlglot
    if not os.path.abspath(sqlglot.__file__).startswith(os.path.abspath(root)):
        raise RuntimeError("wrong sqlglot " + sqlglot.__file__)
    errs = {}
    for name in sorted(x.name for x in pkgutil.walk_packages(sqlglot.__path__, "sqlglot.", onerror=lambda n: None)):
        try:
            importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001
            errs[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
    gc.collect()
    stats = {"opaque": 0, "deep": 0}
    addr = re.compile(r" at 0x[0-9a-fA-F]+")
    memo = {}

    def qn(t):
        return f"{getattr(t, '__module__', '?')}.{getattr(t, '__qualname__', getattr(t, '__name__', '?'))}"

    def cls_ref(o):
        return f"T:{qn(o)}<{','.join(qn(b) for b in o.__bases__)}>"

    def code_fp(c):
        consts = ",".join(code_fp(x) if isinstance(x, types.CodeType) else fp(x, 0, frozenset(), False)
                          for x in c.co_consts)
        return (f"C({c.co_qualname}|{c.co_code.hex()}|{consts}|{c.co_names}|{c.co_varnames}|{c.co_freevars}|"
                f"{c.co_cellvars}|{c.co_argcount},{c.co_posonlyargcount},{c.co_kwonlyargcount},{c.co_flags},"
                f"{c.co_stacksize}|{c.co_exceptiontable.hex()})")

    def fp(o, d, stack, bag):
        t = type(o)
        if o is None or t in (bool, int, float, complex, str, bytes):
            return repr(o)
        if isinstance(o, enum.Enum):
            return f"E:{qn(t)}.{o.name}=" + fp(o._value_, d + 1, stack, bag)
        if isinstance(o, type):
            return cls_ref(o)
        if isinstance(o, types.ModuleType):
            return f"M:{o.__name__}"
        if isinstance(o, types.CodeType):
            return code_fp(o)
        if id(o) in stack:
            return "CYCLE"
        hit = memo.get((id(o), bag))
        if hit is not None and hit[0] is o:
            return hit[1]
        if d > 200:
            stats["deep"] += 1
            return f"DEEP:{qn(t)}"
        stack = stack | {id(o)}

        def r(x):
            return fp(x, d + 1, stack, bag)
        if isinstance(o, (tuple, list)):
            items = [r(x) for x in o]
            out = f"{qn(t)}[" + ",".join(sorted(items) if bag else items) + "]"
        elif isinstance(o, (set, frozenset)):
            out = f"{qn(t)}{{" + ",".join(sorted(r(x) for x in o)) + "}"
        elif isinstance(o, (dict, types.MappingProxyType)):
            items = [r(k) + ":" + r(v) for k, v in o.items()]
            out = f"{qn(t)}{{" + ",".join(sorted(items) if bag else items) + "}"
        elif isinstance(o, types.FunctionType):
            cells = []
            for c in o.__closure__ or ():
                try:
                    cells.append(r(c.cell_contents))
                except ValueError:
                    cells.append("EMPTY")
            out = (f"F({o.__module__}|{o.__qualname__}|{code_fp(o.__code__)}|{r(o.__defaults__)}|"
                   f"{r(o.__kwdefaults__)}|[{','.join(cells)}]|{r(o.__dict__)}|{r(getattr(o, '__annotations__', None))})")
        elif isinstance(o, (staticmethod, classmethod)):
            out = f"{t.__name__}(" + r(o.__func__) + ")"
        elif isinstance(o, property):
            out = "property(" + ",".join(r(f) for f in (o.fget, o.fset, o.fdel)) + ")"
        elif isinstance(o, functools.partial):
            out = "partial(" + r(o.func) + "," + r(o.args) + "," + r(o.keywords) + ")"
        elif isinstance(o, re.Pattern):
            out = f"re({o.pattern!r},{o.flags})"
        elif isinstance(o, types.MethodType):
            out = "meth(" + r(o.__func__) + "," + r(o.__self__) + ")"
        elif isinstance(o, (types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType,
                            types.MethodWrapperType, types.GetSetDescriptorType, types.MemberDescriptorType)):
            out = f"B:{getattr(o, '__qualname__', '?')}:{getattr(getattr(o, '__self__', None), '__name__', '')}"
        else:
            parts = [f"O:{qn(t)}"]
            seen = False
            if hasattr(o, "__wrapped__"):
                parts.append("w=" + r(o.__wrapped__))
                seen = True
            try:
                parts.append(r(object.__getattribute__(o, "__dict__")))
                seen = True
            except (AttributeError, TypeError):
                pass
            for c in t.__mro__:
                sl = c.__dict__.get("__slots__", ())
                for k in ([sl] if isinstance(sl, str) else sl):
                    try:
                        parts.append(f"{k}=" + r(getattr(o, k)))
                    except AttributeError:
                        parts.append(f"{k}=UNSET")
                    seen = True
            if not seen:
                rep = repr(o)
                if addr.search(rep):
                    stats["opaque"] += 1
                parts.append(addr.sub("", rep))
            out = "".join(parts)
        memo[(id(o), bag)] = (o, out)
        return out

    def h(s):
        return hashlib.sha1(s.encode("utf-8", "surrogatepass")).hexdigest()[:20]

    def tag(v):
        if isinstance(v, type):
            return "T"
        if callable(v) or hasattr(type(v), "__get__") or isinstance(v, (staticmethod, classmethod, property)):
            return "F"
        return "V"

    keys, bags, done = {}, {}, set()
    bag_keys = set(bag_keys)

    def put(key, value, t, mod, qual):
        keys[key] = [h(fp(value, 0, frozenset(), False)), t, mod, qual]
        if key in bag_keys:
            bags[key] = h(fp(value, 0, frozenset(), True))

    def do_class(c, path, mod, qual):
        if id(c) in done:
            return
        done.add(id(c))
        keys[path + "::bases"] = [h(",".join(cls_ref(b) for b in c.__bases__)), "S", mod, qual]
        keys[path + "::meta"] = [h(qn(type(c))), "S", mod, qual]
        keys[path + "::order"] = [h(",".join(k for k in c.__dict__ if k not in SNAP_SKIP_CLASS)), "S", mod, qual]
        # Entries carry the class's own module and qualname (what the map's class table is keyed by), not the path
        # the walk reached it through (a per-dialect class made with type() is reached as Dialect.tokenizer_class).
        cmod, cqual = getattr(c, "__module__", "") or "", getattr(c, "__qualname__", "") or ""
        for k, v in c.__dict__.items():
            if k in SNAP_SKIP_CLASS:
                continue
            if isinstance(v, type) and v.__module__.startswith("sqlglot"):
                keys[f"{path}.{k}"] = [h(cls_ref(v)), "T", cmod, cqual]
                do_class(v, f"{path}.{k}", mod, f"{qual}.{k}")
            else:
                put(f"{path}.{k}", v, tag(v), cmod, cqual)

    for n, mo in sorted((n, mo) for n, mo in sys.modules.items() if n == "sqlglot" or n.startswith("sqlglot.")):
        dct = vars(mo)
        keys[n + "::order"] = [h(",".join(k for k in dct if k not in SNAP_SKIP_MOD)), "S", n, ""]
        for k, v in dct.items():
            if k in SNAP_SKIP_MOD:
                continue
            if isinstance(v, type) and v.__module__ == n and v.__qualname__ == k:
                keys[f"{n}:{k}"] = [h(cls_ref(v)), "GT", n, ""]
                do_class(v, f"{n}.{k}", n, k)
            elif isinstance(v, types.ModuleType):
                put(f"{n}:{k}", v, "GM", n, "")
            else:
                put(f"{n}:{k}", v, {"T": "GT", "F": "GF"}.get(tag(v), "G"), n, "")
    for n, e in errs.items():
        keys["IMPORTERROR:" + n] = [h(e), "E", n, ""]
    stats["modules"] = len([n for n in sys.modules if n == "sqlglot" or n.startswith("sqlglot.")])
    stats["errors"] = len(errs)
    return keys, bags, stats


def snap_env():
    """The snapshot child's environment: a fixed hash seed (sets of strings iterate the same in both trees)."""
    return {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": "/tmp", "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1", "SKIP_INTEGRATION": "0"}


def snap_run(root, out, bag_keys=(), pad=0):
    """One full snapshot of root in a fresh process, written to out. pad shifts the process's memory layout (the
    order of sets of classes follows their addresses)."""
    with open(out + ".bag", "w") as fh:
        json.dump(sorted(bag_keys), fh)
    env = dict(snap_env(), BR_SNAP_PAD=str(pad))
    subprocess.run([sys.executable, "-s", "-P", os.path.abspath(__file__), "--snap", root, out, out + ".bag"],
                   env=env, check=True, cwd="/tmp", stdout=subprocess.DEVNULL)
    with open(out) as fh:
        return json.load(fh)


def build_state_base(root, dest):
    """Build time: the pinned tree's import-time state, eight times from copies at paths of different lengths and
    with different memory layouts, to find values whose order or content differs from run to run; those are
    compared as bags (order-free), twice more to make sure the bags agree."""
    import random
    tmp = tempfile.mkdtemp(prefix="sd")
    rng = random.Random(20261122)

    def run(i, bag_keys=()):
        copy_root = os.path.join(tmp, "t" * (1 + rng.randrange(40)) + str(i))
        shutil.copytree(os.path.join(root, "sqlglot"), os.path.join(copy_root, "sqlglot"))
        try:
            return snap_run(copy_root, os.path.join(tmp, f"s{i}.json"), bag_keys, pad=rng.randrange(0, 200000))
        finally:
            shutil.rmtree(copy_root, ignore_errors=True)
    try:
        runs = [run(i) for i in range(8)]
        base = runs[0]["keys"]
        unstable = sorted({k for r in runs[1:] for k in set(base) | set(r["keys"]) if r["keys"].get(k) != base.get(k)})
        b1 = run(8, unstable)
        b2 = run(9, unstable)
        never = sorted(k for k in unstable if b1["bags"].get(k) != b2["bags"].get(k))
        if any(r["stats"]["deep"] for r in runs):
            raise RuntimeError(f"pinned state not comparable: {runs[0]['stats']}")
        with open(dest, "w") as fh:
            json.dump({"keys": base, "unstable": unstable, "bags": b1["bags"], "never": never,
                       "stats": runs[0]["stats"]}, fh)
        print("state base:", len(base), "keys,", len(unstable), "unstable,", len(never), "never comparable,",
              runs[0]["stats"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def snap_main(argv):
    """--snap ROOT OUT BAGFILE: write one snapshot."""
    root, out, bagfile = argv
    with open(bagfile) as fh:
        bag_keys = json.load(fh)
    pad = int(os.environ.get("BR_SNAP_PAD", "0") or 0)
    ballast = [object() for _ in range(pad)], bytearray(pad * 7)  # noqa: F841  (shifts addresses; kept alive)
    keys, bags, stats = snap_state(root, bag_keys)
    with open(out, "w") as fh:
        json.dump({"keys": keys, "bags": bags, "stats": stats}, fh)
    return 0


def state_diff(tmp):
    """The patched tree's import-time differences from the pinned one: [(key, base record, new record)], or a
    string saying why they cannot be compared."""
    if not os.path.isfile(SD):
        return "no state base"
    with open(SD) as fh:
        base = json.load(fh)
    out = os.path.join(tmp, "snap.json")
    try:
        new = snap_run(tmp, out, base["unstable"])
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        return f"snapshot failed: {exc!r}"
    if new["stats"].get("deep") or new["stats"].get("opaque") != base["stats"].get("opaque"):
        return f"snapshot not comparable: {new['stats']} vs {base['stats']}"
    unstable, never = set(base["unstable"]), set(base["never"])
    diffs = []
    for k in sorted(set(base["keys"]) | set(new["keys"])):
        a, b = base["keys"].get(k), new["keys"].get(k)
        if k in never:
            return f"never comparable: {k}"
        if k in unstable:
            if a is None or b is None or base["bags"].get(k) != new["bags"].get(k) or a[1:] != b[1:]:
                diffs.append((k, a, b))
        elif a != b:
            diffs.append((k, a, b))
    return diffs


def nonbody_only(old_text, new_text):
    """True when a change edits code outside function bodies and no function body."""
    try:
        old_t, new_t = ast.parse(old_text), ast.parse(new_text)
    except (SyntaxError, ValueError):
        return False
    of, nf = outer_functions(old_t), outer_functions(new_t)
    if [q for q, _ in of] != [q for q, _ in nf]:
        return False
    for (_, a), (_, b) in zip(of, nf):
        if ast.dump(ast.Module(body=a.body, type_ignores=[])) != ast.dump(ast.Module(body=b.body, type_ignores=[])):
            return False
    for t in (old_t, new_t):
        for _, n in outer_functions(t):
            n.body = [ast.Pass()]
    return ast.dump(old_t) != ast.dump(new_t)


def _stub(text):
    t = ast.parse(text)
    for _, n in outer_functions(t):
        n.body = [ast.Pass()]
    return t


def _blank(t):
    """What v10's check can follow, blanked out: values of `NAME = ...` in class bodies and at module level (also
    inside module-level if/try/with blocks), module-level if tests, def lines (arguments, defaults, decorators,
    return annotations), docstrings."""
    def doc(body):
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and \
                isinstance(body[0].value.value, str):
            body[0].value = ast.Constant("")

    def fn(n):
        n.args = ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None,
                               defaults=[])
        n.decorator_list, n.returns = [], None

    def assign(st):
        if isinstance(st, ast.Assign) and all(isinstance(x, ast.Name) for x in st.targets):
            st.value = ast.Constant(None)
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
            st.value = ast.Constant(None)

    def module_body(body):
        for st in body:
            assign(st)
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn(st)
            elif isinstance(st, ast.ClassDef):
                class_body(st)
            elif isinstance(st, ast.If):
                st.test = ast.Constant(None)
                module_body(st.body)
                module_body(st.orelse)
            elif isinstance(st, (ast.Try, ast.With)):
                for f in ("body", "orelse", "finalbody"):
                    module_body(getattr(st, f, []))
                for hd in getattr(st, "handlers", []):
                    module_body(hd.body)

    def class_body(c):
        doc(c.body)
        for st in c.body:
            assign(st)
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn(st)
            elif isinstance(st, ast.ClassDef):
                class_body(st)
    doc(t.body)
    module_body(t.body)
    return t


def import_code(st, mod, is_pkg):
    """An import statement as absolute source code (None: not an import)."""
    if isinstance(st, ast.Import):
        return "import " + ", ".join(a.name for a in st.names)
    if isinstance(st, ast.ImportFrom):
        base = st.module or ""
        if st.level:
            pkg = mod.split(".") if is_pkg else mod.split(".")[:-1]
            pkg = pkg[: len(pkg) - (st.level - 1)]
            base = ".".join(pkg + ([st.module] if st.module else []))
        return f"from {base} import " + ", ".join(a.name for a in st.names)
    return None


def static_items(old_text, new_text, mod, is_pkg=False):
    """None when, with function bodies left out, the change edits something v10's check cannot follow (a statement
    added, removed or replaced, class bases or decorators, ...). Else the edited names:
    ("c", module, class qualname, name) for class entries, ("m", module, name) for module globals."""
    try:
        a, b = _stub(old_text), _stub(new_text)
        if ast.dump(_blank(_stub(old_text))) != ast.dump(_blank(_stub(new_text))):
            return None
    except (SyntaxError, ValueError):
        return None
    items = []

    def bound(body):
        out = set()
        for st in body:
            for n in ast.walk(st):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.add(n.id)
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    out.add(n.name)
                elif isinstance(n, ast.alias):
                    out.add((n.asname or n.name).split(".")[0])
        return out

    def names_of(st):
        if isinstance(st, ast.Assign):
            return [x.id for x in st.targets]
        if isinstance(st, ast.AnnAssign):
            return [st.target.id]
        return [st.name]

    def walk(x, y, qual):
        for p, q in zip(x, y):
            if ast.dump(p) == ast.dump(q):
                continue
            if isinstance(p, ast.ClassDef):
                walk(p.body, q.body, f"{qual}.{p.name}" if qual else p.name)
            elif isinstance(p, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.AsyncFunctionDef)):
                for nm in names_of(p):
                    items.append(("c", mod, qual, nm) if qual else ("m", mod, nm))
            elif isinstance(p, ast.Expr):
                continue  # a docstring
            elif isinstance(p, ast.If) and not qual:
                if ast.dump(p.test) != ast.dump(q.test):
                    items.extend(("a", mod, nm) for nm in sorted(bound(p.body) | bound(p.orelse)))
                    for st in [x for blk in (p.body, p.orelse) for n in blk for x in ast.walk(n)]:
                        code = import_code(st, mod, is_pkg)
                        if code:
                            items.append(("i", code))
                walk(p.body, q.body, qual)
                walk(p.orelse, q.orelse, qual)
            elif isinstance(p, (ast.Try, ast.With)) and not qual:
                for f in ("body", "orelse", "finalbody"):
                    walk(getattr(p, f, []), getattr(q, f, []), qual)
                for hp, hq in zip(getattr(p, "handlers", []), getattr(q, "handlers", [])):
                    walk(hp.body, hq.body, qual)
            else:
                items.append(("?", mod, qual, type(p).__name__))
    walk(a.body, b.body, "")
    return None if any(i[0] == "?" for i in items) else items


def name_masks(tq, item):
    """Stage masks of the lines test_execute_tpcds runs (inside functions) that can read this edited name, or a
    reason to refuse. Class entries: only if the test reads that class's attribute at run time (map class table),
    then every line reading `.NAME` or the string "NAME". Module globals: that module's lines reading the bare NAME,
    and `.NAME` / "NAME" anywhere (module attribute reads, getattr by literal)."""
    names, cls, tnames, bnames = tq["names"], tq["cls"], set(tq["tnames"]), tq.get("bnames", {})
    if item[0] == "c":
        _, mod, qual, name = item
        if not qual or name.startswith("__") or name.endswith("_sql") or name in tnames:
            return f"attribute the check cannot follow: {qual}.{name}"
        if cls.get(f"{mod}|{qual}|{name}") == 0:
            return [0] * 6
        return list(names.get(name) or [0] * 6)
    _, mod, name = item
    if name.startswith("__") or name in tnames:
        return f"global the check cannot follow: {mod}.{name}"
    if "anames" not in tq:  # an older map: every `.NAME` read anywhere
        a, b = names.get(name) or [0] * 6, bnames.get(f"{mod}|{name}") or [0] * 6
        return [x | y for x, y in zip(a, b)]
    # The module's own lines reading the bare name; `X.NAME` where X is a name the module is imported under (or a
    # package alias plus the rest of its dotted name); `.NAME` on anything that is not a chain of names; "NAME".
    an, malias = tq["anames"], tq["malias"]
    chains = set(malias.get(mod, [mod]))
    parts = mod.split(".")
    for i in range(1, len(parts)):
        rest = ".".join(parts[i:])
        chains |= {f"{al}.{rest}" for al in malias.get(".".join(parts[:i]), [])}
    rows = [bnames.get(f"{mod}|{name}"), tq["snames"].get(name), an.get(f"?|{name}")] +         [an.get(f"{c}|{name}") for c in chains]
    out = [0] * 6
    for r in rows:
        if r:
            out = [x | y for x, y in zip(out, r)]
    return out


def mask_est(tq, rows):
    """CPU seconds of the queries these stage masks need (0.0: none), or a reason to refuse."""
    p_m = o_m = g_m = d_m = e_m = 0
    for r in rows:
        if isinstance(r, str):
            return r
        p, o, g, d, st, e = r
        if st:
            return "read while the tables are built"
        p_m, o_m, g_m, d_m, e_m = p_m | p, o_m | o, g_m | g, d_m | d, e_m | e
    q1 = (p_m & tq["exec_mask"]) | o_m | g_m
    if (q1 | d_m) & tq["unstable_mask"]:
        return "unstable query"
    if not (q1 | d_m | e_m):
        return 0.0
    qs = tq["q"]
    return 1.0 + sum(qs[i]["cpu"] for i in range(len(qs)) if (q1 | d_m) >> i & 1) + \
        sum(qs[i]["xcpu"] for i in range(len(qs)) if e_m >> i & 1)


def cls_job(m, patch_text, tp):
    """The class-level check for a change that edits only code outside function bodies, or None."""
    tq = m.d.get("tpq")
    if not tq or "names" not in tq or "cls" not in tq or not os.path.isfile(SD):
        return None
    entries = list(parse_patch(patch_text))
    for old, new, blocks in entries:  # path rules first: they cost nothing, parsing a big file does
        path = old or new
        if not path:
            return None
        if path.startswith("sqlglot/"):
            if not (old and new == old and path.endswith(".py") and path in m.d["files"]):
                return None
            continue
        for p in {old, new} - {None, ""}:
            if p.endswith(".py") or p.startswith(TPCDS_DEPS) or p in CONFIG_FILES or "/" not in p or p in TPC_FIX:
                return None
            if p in m.d["opens"] and m.set(m.d["opens"][p]) >> tp & 1:
                return None
        if old != new or old is None:
            return None
    files, nonbody, triples, probes = {}, False, [], []
    for old, new, blocks in entries:
        path = old
        if not path.startswith("sqlglot/"):
            continue
        text = m.source(path)
        new_text = patched(text, blocks) if text is not None else None
        if new_text is None:
            return None
        if not nonbody_only(text, new_text):
            if changed_bodies(text, new_text) != []:
                return None  # a function body (or something unparsable) changed: not this check's case
        else:
            mod = path[:-3].replace("/", ".")
            mod = mod[: -len(".__init__")] if mod.endswith(".__init__") else mod
            its = static_items(text, new_text, mod, path.endswith("/__init__.py"))
            if its is None:
                return None
            probes += [i[1] for i in its if i[0] == "i"]
            if any(i[0] == "i" for i in its):
                probes.append(f"import {mod}")
            triples += [i for i in its if i[0] in ("c", "m")]  # flipped-if names: the state check decides
            nonbody = True
        files[path] = new_text
    if not files or not nonbody:
        return None
    q_est = mask_est(tq, [name_masks(tq, it) for it in triples])
    if isinstance(q_est, str) or q_est > CHECK_MAX_EST_S:
        return None  # the check would refuse, or its queries alone would cost more than one check may
    return {"kind": "cls", "files": files, "est": SNAP_EST_S + q_est + PROBE_EST_S * len(set(probes)),
            "probe": sorted(set(probes))}


def cls_check():
    """Child process: stdin is a cls_job. Prints "same" last when test_execute_tpcds provably cannot change."""
    job = json.loads(sys.stdin.read())
    sys.dont_write_bytecode = True
    t_start = time.process_time()
    tmp = patched_tree(job["files"])
    try:
        for code in job.get("probe", []):
            # Each statement first thing after `import sqlglot`, in a fresh process: an import that a flipped
            # TYPE_CHECKING block now runs can close a cycle that only some import orders hit.
            pr = subprocess.run([sys.executable, "-s", "-P", "-c",
                                 "import sys; sys.path.insert(0, sys.argv[1]); import sqlglot; exec(sys.argv[2])",
                                 tmp, code], env=snap_env(), cwd="/tmp", capture_output=True, text=True, timeout=60)
            if pr.returncode:
                print("import probe failed:", code, pr.stderr.strip().splitlines()[-1:] or "")
                return 1
        diffs = state_diff(tmp)
        if isinstance(diffs, str):
            print(diffs)
            return 1
        m = Map(MAP)
        tq = m.d["tpq"]
        builtin_names = set(dir(__import__("builtins")))
        p_m = o_m = g_m = d_m = e_m = 0
        seen = []
        for k, a, b in diffs:
            rec = b or a
            t, mod, qual = rec[1], rec[2], rec[3]
            if k.startswith("IMPORTERROR"):
                print("import error:", k)
                return 1
            if k.endswith("::order") and t == "S" and not qual and k[: -len("::order")] == mod:
                continue  # the order of a module's globals (names added or moved): nothing reads it at run time
            if "::" in k:
                print("class structure changed:", k)
                return 1
            if qual and t in ("V", "F"):  # a class dict entry: a value, a method or a descriptor
                if a is None or b is None or a[1] != b[1]:
                    print("class entry added, removed or changed kind:", k)
                    return 1
                item = ("c", mod, qual, k.rsplit(".", 1)[1])
            elif not qual and t in ("G", "GF", "GT", "GM") and k.startswith(mod + ":"):
                name = k[len(mod) + 1:]
                if a is None:
                    # A global that did not exist (a TYPE_CHECKING import now made): no code could read it before
                    # without failing, unless the name is a builtin it now shadows.
                    if name in builtin_names or name.startswith("__"):
                        print("new global shadows a builtin:", k)
                        return 1
                    seen.append(f"+{mod}.{name}")
                    continue
                item = ("m", mod, name)
            else:
                print("not a followed kind:", k, a and a[1], b and b[1])
                return 1
            r = name_masks(tq, item)
            if isinstance(r, str):
                print(r)
                return 1
            seen.append(".".join(x for x in item[1:] if x))
            p, o, g, d, st, e = r
            if st:
                print("read while the tables are built:", k)
                return 1
            p_m, o_m, g_m, d_m, e_m = p_m | p, o_m | o, g_m | g, d_m | d, e_m | e
        ex = tq["exec_mask"]
        q1 = (p_m & ex) | o_m | g_m
        pn = p_m & ~ex
        if (q1 | d_m) & tq["unstable_mask"]:
            print("unstable query")
            return 1
        if not (q1 | pn | d_m | e_m):
            print("changed:", seen)
            print("same")
            return 0
        if e_m and not tables_ready():
            print("no tables")
            return 1
        qs = tq["q"]
        est = 1.0 + sum(qs[i]["cpu"] for i in range(len(qs)) if (q1 | d_m) >> i & 1) + \
            sum(qs[i]["xcpu"] for i in range(len(qs)) if e_m >> i & 1)
        left = CHECK_MAX_EST_S - (time.process_time() - t_start)
        if est > left:
            print(f"queries would cost {est:.1f} s, {left:.1f} s left")
            return 1
        pick = lambda mask, keys: [{"i": i, **{k: qs[i][k] for k in keys}} for i in range(len(qs)) if mask >> i & 1]  # noqa: E731
        qjob = {"files": job["files"], "schema": tq["schema"], "est": est, "q1": pick(q1, ("sql", "fo", "res", "xcpu")),
                "pn": pick(pn, ("sql",)), "d": pick(d_m, ("sql", "duck")), "px": [],
                "xq": pick(e_m, ("sql", "res", "xcpu")), "xbudget": round(max(0.0, left - est), 3)}
        print("changed:", seen)
        return query_check(qjob, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_checks(jobs, t0):
    """{commit: True when the check proved the test unaffected}. Two at a time; none starts after the budget."""
    def one(k):
        if time.time() - t0 > CHECK_BUDGET_S:
            return k, False
        job = jobs[k]
        cls = job.get("kind") == "cls"
        try:
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--cls-check" if cls else "--tpcds-check"],
                               input=json.dumps(job), capture_output=True, text=True,
                               timeout=4 * CHECK_MAX_EST_S if cls else max(20.0, 4 * (job["est"] + job.get("xbudget", 0.0))))
        except subprocess.TimeoutExpired:
            print(f"{k}: tpcds check timed out", file=sys.stderr)
            return k, False
        lines = p.stdout.strip().splitlines()
        ok = p.returncode == 0 and lines[-1:] == ["same"]
        if not ok:
            print(f"{k}: tpcds check keeps the test: rc {p.returncode} {lines[-1:]} {p.stderr[-300:]!r}",
                  file=sys.stderr)
        return k, ok

    if not jobs:
        return {}
    with ThreadPoolExecutor(CHECK_JOBS) as ex:
        return dict(ex.map(one, sorted(jobs, key=lambda k: jobs[k]["est"])))


def prepare(dest):
    """Build time: write the pinned sqlglot tree from the map and compile it, for the checks."""
    import compileall
    m = Map(MAP)
    for path, b in m.d["src"].items():
        if path.startswith("sqlglot/"):
            p = os.path.join(dest, path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(m.source(path))
    ok = compileall.compile_dir(os.path.join(dest, "sqlglot"), quiet=1)
    print("prepared", dest, "compiled" if ok else "COMPILE ERRORS")
    try:
        build_tables(m, dest, TP)
    except Exception as exc:  # noqa: BLE001  (no tables: checks that must execute a query keep the test)
        print("TPC-DS tables not built:", repr(exc))
    try:
        build_state_base(dest, SD)
    except Exception as exc:  # noqa: BLE001  (no state base: the class-level check never runs)
        print("state base not built:", repr(exc))
        if os.path.exists(SD):
            os.remove(SD)


def build_tables(m, dest, out):
    """Build time: the TPC-DS tables exactly as TestExecutor.setUpClass builds them (python side), from the csv files
    in the map, with the pinned sqlglot; one pickle per table. Every table must hash as the test's own did, else none
    is kept."""
    import ast as pyast
    import csv
    import io
    if "tpcsv" not in m.d:
        return
    sys.path.insert(0, dest)
    from sqlglot.executor.table import Table
    if not os.path.abspath(sys.modules["sqlglot"].__file__).startswith(os.path.abspath(dest)):
        raise RuntimeError("wrong sqlglot")
    os.makedirs(out, exist_ok=True)
    for table, columns in m.d["tpq"]["schema"].items():
        raw = base64.b64decode(m.d["tpcsv"][table])
        if raw[:2] != b"\x1f\x8b":
            raise RuntimeError(f"{table}: not gzip")
        reader = csv.reader(gzip.open(io.BytesIO(raw), "rt", newline=""), delimiter="|")  # open_file in the test
        rows = []
        ctypes = []
        next(reader)
        for row in reader:
            if not ctypes:
                for i, v in enumerate(row):
                    try:
                        ctypes.append(type(pyast.literal_eval(v)))
                    except (ValueError, SyntaxError):
                        ctypes.append(str)
            rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
        t = pickle.loads(pickle.dumps(Table(columns=columns, rows=rows)))  # the initializer receives them pickled
        h = hashlib.sha1(repr((list(t.columns), t.rows)).encode("utf-8", "surrogatepass")).hexdigest()
        if h != m.d["tphash"][table]:
            raise RuntimeError(f"{table}: built table differs from the test's")
        with open(os.path.join(out, table + ".pkl"), "wb") as fh:
            pickle.dump(t, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out, "ok"), "w") as fh:
        fh.write("ok\n")
    print("built", len(m.d["tpq"]["schema"]), "TPC-DS tables in", out)


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
    t0 = time.time()
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    ms = {t["number"]: round(t["seconds"] * 1000) for t in tests}
    tp = next((t["number"] for t in tests if t["id"] == TPCDS_ID), None)
    m = Map(MAP)
    tx = m.d.get("texec")
    if tx:  # the narrowing names tests by number: only when this run numbers them as the map does
        ids = {str(t["number"]): t["id"] for t in tests}
        if any(ids.get(k) != v for k, v in tx["ids"].items()) or \
                sum(1 for v in ids.values() if v.startswith("tests/test_executor.py::")) != len(tx["ids"]):
            m.d["texec"] = None
    rows, jobs = [], {}
    for patch in sorted((DATA / "commits").glob("*.patch")):
        text = patch.read_text(encoding="utf-8", errors="replace")
        try:
            mask = select(m, text)
        except Exception as exc:  # never crash a run: run everything for this commit
            print(f"{patch.stem}: {exc!r}", file=sys.stderr)
            mask = m.all_tests()
        chosen = [n for n in ms if mask >> n & 1]
        rows.append((patch.stem, chosen))
        if tp is not None and mask >> tp & 1:
            try:
                job = tpcds_job(m, text, tp)
            except Exception as exc:  # noqa: BLE001  (no check: keep the test)
                print(f"{patch.stem}: tpcds job: {exc!r}", file=sys.stderr)
                job = None
            if job is None:
                try:
                    rest = [n for n in chosen if n != tp]
                    cost = lambda ts: (max(sum(ms[t] for t in w) for w in pack(ts, ms)) + WORKER_MS * len(pack(ts, ms))) / 1000 if ts else 0  # noqa: E731
                    gain = cost(chosen) - cost(rest)
                    job = cls_job(m, text, tp) if gain >= SNAP_EST_S + 2 else None
                    if job is not None and gain < job["est"] + 2:
                        job = None  # the check would cost about what dropping the test saves
                except Exception as exc:  # noqa: BLE001
                    print(f"{patch.stem}: class job: {exc!r}", file=sys.stderr)
                    job = None
            if job is not None:
                jobs[patch.stem] = job
    try:
        dropped = run_checks(jobs, t0)
    except Exception as exc:  # noqa: BLE001
        print(f"tpcds checks: {exc!r}", file=sys.stderr)
        dropped = {}
    for stem, chosen in rows:
        if dropped.get(stem):
            chosen = [n for n in chosen if n != tp]
        print(json.dumps({"commit": stem, "workers": pack(chosen, ms)}))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--prepare"]:
        prepare(sys.argv[2])
    elif sys.argv[1:2] == ["--tpcds-check"]:
        sys.exit(tpcds_check())
    elif sys.argv[1:2] == ["--cls-check"]:
        sys.exit(cls_check())
    elif sys.argv[1:2] == ["--snap"]:
        sys.exit(snap_main(sys.argv[2:5]))
    else:
        main()
