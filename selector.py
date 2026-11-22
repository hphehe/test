"""Pick the tests each change can break, from a map of which tests execute which lines of sqlglot."""

import ast
import base64
import copy
import gzip
import hashlib
import json
import os
import types
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

# The image may run on a second build of the same CPython (clang, see the Dockerfile). Tests that use sqlite3 must then
# see the pinned image's SQLite: its own _sqlite3 extension module is loaded instead of the one that build carries.
IMAGE_SQLITE = "/usr/local/lib/python3.13/lib-dynload/_sqlite3.cpython-313-x86_64-linux-gnu.so"


class _ImageSqlite:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name != "_sqlite3":
            return None
        from importlib.machinery import ExtensionFileLoader, ModuleSpec
        spec = ModuleSpec(name, ExtensionFileLoader(name, IMAGE_SQLITE), origin=IMAGE_SQLITE)
        spec.has_location = True
        return spec


if not os.path.realpath(sys.executable).startswith("/usr/local/") and os.path.isfile(IMAGE_SQLITE):
    sys.meta_path.insert(0, _ImageSqlite)
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
        self.d = None
        pk = str(path) + ".pkl"  # written by --prepare: the same map, read ~5x faster
        try:
            if os.path.getmtime(pk) >= os.path.getmtime(path):
                with open(pk, "rb") as fh:
                    self.d = pickle.load(fh)
        except Exception:  # noqa: BLE001
            self.d = None
        if not isinstance(self.d, dict):
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

    def lams(self, path):
        """[[lineno, col, end_lineno, body_lineno, body_end_lineno, sid, load]] for the lambdas written outside
        functions (see lambda_price)."""
        key = (path, "lam")
        if key not in self._blocks:
            self._blocks[key] = json.loads(self.d["files"][path].get("lam", "[]"))
        return self._blocks[key]

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
        got = dropped_decorator(m, path, removed, added)
        if got is not None:
            sel |= got
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


_DESCRIPTORS = ("property", "classmethod", "staticmethod")


def plain_wrapper(m, tree, name):
    """True when `name` is a decorator written at the top of this module (`tree`) as
        def name(func):  ["doc"]  [@wraps(func)] def inner(...): <first statement calls func> ...  return inner
    : applying it does nothing else, and every call of what it made runs the function it was given."""
    d = next((s for s in tree.body if isinstance(s, ast.FunctionDef) and s.name == name), None)
    if d is None or d.decorator_list or len(d.args.args) != 1 or d.args.vararg or d.args.kwarg or \
            d.args.kwonlyargs or d.args.posonlyargs or d.args.defaults:
        return False
    func = d.args.args[0].arg
    body = d.body[1:] if d.body and isinstance(d.body[0], ast.Expr) and isinstance(
        getattr(d.body[0], "value", None), ast.Constant) else d.body
    if len(body) != 2 or not isinstance(body[0], ast.FunctionDef) or not isinstance(body[1], ast.Return) or \
            not isinstance(body[1].value, ast.Name) or body[1].value.id != body[0].name:
        return False
    inner = body[0]
    for dec in inner.decorator_list:
        f = dec.func if isinstance(dec, ast.Call) else None
        if not (f is not None and ((isinstance(f, ast.Name) and f.id == "wraps") or (
                isinstance(f, ast.Attribute) and f.attr == "wraps" and isinstance(f.value, ast.Name)
                and f.value.id == "functools"))):
            return False
    if not inner.body:
        return False
    first = inner.body[0]
    if not isinstance(first, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr)):
        return False
    v = first.value  # the call itself, not one inside a condition
    return isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == func and all(
        isinstance(a, (ast.Name, ast.Starred)) for a in v.args) and all(
        isinstance(k.value, ast.Name) for k in v.keywords)


def dropped_decorator(m, path, removed, added):
    """For a block that only deletes `@property` / `@classmethod` / `@staticmethod` lines of methods defined in a
    class body: the tests that ran those methods' bodies (any use of such a method runs its body, before and after the
    change), or None. Not for a method whose name decorates another (`@x.setter`), a dunder, or a body that ran
    while the module loaded."""
    if added or not removed or any(not re.fullmatch(r"@\w+", t.strip()) for _, t in removed):
        return None
    src = m.source(path)
    if src is None:
        return None
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    passes = {d for d in _DESCRIPTORS}
    for _, t in removed:
        if t.strip()[1:] not in passes and not plain_wrapper(m, tree, t.strip()[1:]):
            return None
        passes.add(t.strip()[1:])
    lines = {x for x, _ in removed}
    rt, imp = m.spans(path, "rt"), m.spans(path, "imp")
    sel, found = 0, set()
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        fns = [s for s in cls.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        named = {d.value.id for f in fns for d in f.decorator_list
                 if isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name)}
        for fn in fns:
            for d in fn.decorator_list:
                if d.lineno not in lines:
                    continue
                if not (isinstance(d, ast.Name) and d.id in passes) or fn.name in named or \
                        (fn.name.startswith("__") and fn.name.endswith("__")):
                    return None
                found.add(d.lineno)
                lo = fn.body[0].lineno
                for x in range(lo, fn.end_lineno + 1):
                    i = imp.find(x)
                    if i is not None and i[0] >= lo:
                        return None
                    r = rt.find(x)
                    if r is not None:
                        sel |= m.set(r[2])
    return sel if found == lines else None


def outer_lambdas(tree):
    """Lambdas written outside any function, in source order (a lambda inside one belongs to the outer one)."""
    out = []

    def visit(node, in_func):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(ch, True)
            elif isinstance(ch, ast.Lambda):
                if not in_func:
                    out.append(ch)
                visit(ch, True)
            else:
                visit(ch, in_func)
    visit(tree, False)
    return out


def lambda_price(m, path, blocks):
    """Tests for an edit confined to the bodies of lambdas written outside functions (dispatch tables such as
    TRANSFORMS): a lambda's body runs only when something calls it, and the callers do not change, so a test that never
    ran a line of the old body (traced test by test, each in its own process) runs the same code on the new one.
    None (price as before) when anything else changes, a changed lambda ran while a module or class body was loading,
    or a body yields or awaits (a call would not run it)."""
    rows = m.lams(path)
    if not rows:
        return None
    spans = [(r[3], r[4]) for r in rows]
    for removed, added, _ in blocks:
        if not removed or len(removed) != len(added):
            return None  # lines added or removed: positions move
        for ln, _t in removed:
            if not any(s <= ln <= e for s, e in spans):
                return None
    old = m.source(path)
    new = patched(old, blocks) if old is not None else None
    if new is None:
        return None
    try:
        a, b = ast.parse(old), ast.parse(new)
    except (SyntaxError, ValueError):
        return None
    la, lb = outer_lambdas(a), outer_lambdas(b)
    if len(la) != len(lb):
        return None
    by_pos = {(r[0], r[1]): r for r in rows}
    sel, changed = 0, False
    for x, y in zip(la, lb):
        if (x.lineno, x.end_lineno, x.body.lineno, x.body.end_lineno) != \
                (y.lineno, y.end_lineno, y.body.lineno, y.body.end_lineno):
            return None
        if ast.dump(x.body) == ast.dump(y.body):
            continue
        r = by_pos.get((x.lineno, x.col_offset))
        if r is None or r[6]:
            return None
        if any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await)) for t in (x.body, y.body) for n in ast.walk(t)):
            return None
        sel |= m.set(r[5])
        changed = True
    if not changed:
        return None
    for t in (la, lb):
        for lam in t:
            lam.body = ast.Constant(None)
    if ast.dump(a) != ast.dump(b):
        return None  # something outside the bodies changed too
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


_CODE = {}  # (file name, text) -> its compiled module code, or the SyntaxError / ValueError it raised


def compiled(text, fn):
    """compile(text, fn, "exec", dont_inherit=True), once per text and file name in this process."""
    k = (fn, text)
    r = _CODE.get(k)
    if r is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = compile(text, fn, "exec", dont_inherit=True)
            except (SyntaxError, ValueError) as exc:
                r = exc
        _CODE[k] = r
    if isinstance(r, BaseException):
        raise r
    return r


def breaks_syntax(m, old, new, blocks):
    """True when the changed Python file no longer compiles. Python is the run's version (the image pins 3.13)."""
    if old is None:
        text = "\n".join(t for b in blocks for t in b[1]) + "\n"  # a new file is all added lines
    else:
        src = m.source(old)
        text = patched(src, blocks) if src is not None else None
        if text is None:
            return False
    try:
        compiled(text, os.path.join(ST, new or old))
    except (SyntaxError, ValueError):
        return True
    except Exception:  # noqa: BLE001  (recursion and memory limits of a huge expression: not our concern)
        return False
    return False


_ANN_LINE = re.compile(r"^\s*[A-Za-z_][\w.]*(\[[^\]]*\])?\s*:(?!=)\s*\S")
_BLOCK_WORDS = ("def ", "async ", "class ", "if ", "elif ", "else", "try", "except", "finally", "with ", "for ",
                "while ", "match ", "case ", "return", "lambda", "@")
_ODD_BASES = {"NamedTuple", "TypedDict", "Protocol", "Generic", "BaseModel", "Struct"}


def annotations_only(m, path, blocks):
    """True when an in-place edit of a sqlglot module changes nothing but the annotations of NAME: T (= ...)
    statements that are never evaluated: inside functions (local annotations are not evaluated at all), and in the
    module body or the bodies of undecorated plain classes when the module has `from __future__ import annotations`
    (strings in __annotations__, which sqlglot reads only through its dataclasses)."""
    lines = [t for b in blocks for _, t in b[0]] + [t for b in blocks for t in b[1]]
    if not lines or any(len(b[0]) != len(b[1]) for b in blocks):
        return False
    for t in lines:
        st = t.lstrip()
        if not _ANN_LINE.match(t) or st.startswith(_BLOCK_WORDS):
            return False
    src = m.source(path)
    new = patched(src, blocks) if src is not None else None
    if new is None:
        return False
    try:
        a, b = ast.parse(src), ast.parse(new)
    except (SyntaxError, ValueError):
        return False

    def future(t):
        return any(isinstance(n, ast.ImportFrom) and n.module == "__future__" and any(x.name == "annotations"
                                                                                       for x in n.names) for n in t.body)
    deferred = future(a) and future(b)

    def blank(body, in_def):
        for st in body:
            if isinstance(st, ast.AnnAssign) and (in_def or deferred):
                st.annotation = ast.Constant(None)
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                blank(st.body, True)
            elif isinstance(st, ast.ClassDef):
                plain = not st.decorator_list and not st.keywords and not any(
                    (x.id if isinstance(x, ast.Name) else x.attr if isinstance(x, ast.Attribute) else "?") in _ODD_BASES
                    or not isinstance(x, (ast.Name, ast.Attribute)) for x in st.bases)
                if plain:
                    blank(st.body, False)
            for f in ("orelse", "finalbody"):
                if not isinstance(st, ast.ClassDef) and isinstance(getattr(st, f, None), list):
                    blank(getattr(st, f), in_def)
            if isinstance(st, (ast.If, ast.Try, ast.With, ast.For, ast.While, ast.AsyncFor, ast.AsyncWith)):
                blank(st.body, in_def)
            for hd in getattr(st, "handlers", []):
                blank(hd.body, in_def)
    if ast.dump(a) == ast.dump(b):
        return False  # no change in the code at all: the other rules see it
    blank(a.body, False)
    blank(b.body, False)
    return ast.dump(a) == ast.dump(b)


def _checking(test):
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or \
        (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING" and isinstance(test.value, ast.Name))


_CHECK_IF = re.compile(r"^[ \t]*if[ \t]+(?:\w+\.)?TYPE_CHECKING[ \t]*:[ \t]*(?:#.*)?$", re.M)


def checking_only(m, path, blocks):
    """True when an in-place edit of a sqlglot module changes only the bodies of `if TYPE_CHECKING:` blocks (typing's
    flag, False at run time: those bodies never run; an `else:` branch does and stays compared)."""
    # cheap first: such a body never runs, so none of its lines is in a run span
    rt = m.spans(path, "rt")
    for removed, added, at in blocks:
        for x in [x for x, _ in removed] or [at - 1]:
            if rt.find(x) is not None:
                return False
    src = m.source(path)
    if src is None or "TYPE_CHECKING" not in src:
        return False
    # then every changed line inside such a block by indentation (the trees below decide)
    lines = src.split("\n")
    spans = []
    for mt in _CHECK_IF.finditer(src):
        h = src.count("\n", 0, mt.start()) + 1
        c, end = indent(lines[h - 1]), h
        for j in range(h, len(lines)):
            u = lines[j]
            if u.strip() and indent(u) <= c:
                break
            end = j + 1
        spans.append((h, end, c))
    for removed, added, at in blocks:
        if not any(all(h < x <= e for x, _ in removed) and h < at <= e + 1 and
                   all(not t.strip() or indent(t) > c for t in added) for h, e, c in spans):
            return False
    new = patched(src, blocks)
    if new is None:
        return False
    try:
        a, b = ast.parse(src), ast.parse(new)
    except (SyntaxError, ValueError):
        return False
    if ast.dump(a) == ast.dump(b):
        return False

    def blank(tree):
        n = 0
        for st in ast.walk(tree):
            if isinstance(st, ast.If) and _checking(st.test):
                st.body = [ast.Pass()]
                n += 1
        return n
    return bool(blank(a)) and bool(blank(b)) and ast.dump(a) == ast.dump(b)


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
            if annotations_only(m, path, blocks) or checking_only(m, path, blocks):
                continue  # (the tests that read the file as text are in already)
            lam = lambda_price(m, path, blocks)
            sel |= lam if lam is not None else select_sqlglot(m, path, blocks)
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
# Queries whose optimized tree differs are executed and their results compared. They may cost CPU in proportion to
# the seconds that leaving test_execute_tpcds out saves (the board charges about 0.6-0.8 of a local CPU second).
CHECK_X_PER_S = float(os.environ.get("BR_CHECK_X_PER_S", 0.9))
CHECK_X_CAP_S = float(os.environ.get("BR_CHECK_X_CAP", 20))
CHECK_X_INLINE_S = 0.5  # a differing query this cheap runs as soon as it is found
CHECK_EST_PER_S = float(os.environ.get("BR_CHECK_EST_PER_S", 1.5))  # estimated CPU a check may cost per second saved
CHECK_EST_CAP_S = float(os.environ.get("BR_CHECK_EST_CAP", 36))


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
    if ast.dump(old_t) == ast.dump(new_t):
        return spans
    # Outside function bodies only the bodies of lambdas may differ (an entry of a table of lambdas): the new value
    # is a function whose body runs only where it is called, as a function body does. The two trees pair their
    # lambdas in walk order once the bodies are set aside.
    lo = [n for n in ast.walk(old_t) if isinstance(n, ast.Lambda)]
    ln = [n for n in ast.walk(new_t) if isinstance(n, ast.Lambda)]
    if len(lo) != len(ln):
        return None
    bodies = [(a, b, ast.dump(a.body) != ast.dump(b.body)) for a, b in zip(lo, ln)]
    for a, b, _ in bodies:
        a.body = b.body = ast.Constant(None)
    if ast.dump(old_t) != ast.dump(new_t):
        return None
    for a, b, diff in bodies:
        if diff:
            spans.append((a.lineno, a.end_lineno, "lambda"))
    return spans


def line_span(m, path, ln):
    """(start, end, mask) of the statement span holding line ln in the map's runtime rows, or None."""
    for s, e, sid in m.d["files"][path]["rt"]:
        if s <= ln <= e:
            return s, e, m.set(sid)
    return None


def tpcds_job(m, patch_text, tp, max_est=None):
    """The check that can drop test_execute_tpcds for this change, or None when it cannot apply (or would cost more
    than max_est CPU seconds, CHECK_MAX_EST_S by default)."""
    max_est = CHECK_MAX_EST_S if max_est is None else max_est
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
            spans += [(path, x[0], x[1], len(x) > 2) for x in sp]
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
    for path, a, b, lam in spans:
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
            if x and not lam:
                return None  # runs at import or in table setup
            # (a lambda's line also runs at import, where the function is made; its body runs where it is called)
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
    q1 &= ~e_m  # a query that is executed is optimized there too (execute() runs the patched optimizer)
    qs = tq["q"]
    est = 1.0 + 0.05 * len(px) + sum(qs[i]["cpu"] for i in range(len(qs)) if (q1 | d_m) >> i & 1) + \
        sum(qs[i]["xcpu"] for i in range(len(qs)) if e_m >> i & 1)
    if est > max_est:
        return None
    pick = lambda mask, keys: [{"i": i, **{k: qs[i][k] for k in keys}} for i in range(len(qs)) if mask >> i & 1]  # noqa: E731
    return {"files": files, "schema": tq["schema"], "est": est, "q1": pick(q1, ("sql", "fo", "res", "xcpu")),
            "pn": pick(pn, ("sql",)), "d": pick(d_m, ("sql", "duck")), "px": px,
            "xq": pick(e_m, ("sql", "res", "xcpu")),
            # CPU left for executing queries whose optimized tree turns out different (only known in the check)
            "xbudget": round(max_est - est, 3) if tables_ready() else 0.0}


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


def patched_tree(files, pyc=False):
    """A temporary copy of the pinned tree with these files replaced. The caller removes it. With pyc, each replaced
    file also gets a .pyc, so that it loads as the pinned files do (a set constant read back from a .pyc can iterate
    in another order than the one the compiler built)."""
    tmp = tempfile.mkdtemp(prefix="tpc")
    shutil.copytree(os.path.join(SQ, "sqlglot"), os.path.join(tmp, "sqlglot"))  # copy2: the .pyc stay valid
    for path, text in files.items():
        with open(os.path.join(tmp, path), "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        # A .pyc that is not checked against its source (unchecked-hash) would still load the pinned code.
        d, base = os.path.split(os.path.join(tmp, path))
        cache = os.path.join(d, "__pycache__")
        if os.path.isdir(cache):
            for f in os.listdir(cache):
                if f.startswith(base[:-3] + ".") and f.endswith(".pyc"):
                    os.remove(os.path.join(cache, f))
        if pyc and path.endswith(".py"):
            import py_compile
            try:
                py_compile.compile(os.path.join(tmp, path), doraise=True,
                                   invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
            except Exception:  # noqa: BLE001  (the import then compiles it, and reports the error)
                pass
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


DUCK_MAX_TEXTS = 60
_DUCK = {}


def duck_tables(schema):
    """A DuckDB connection (one thread) over the rows the test's views read, loaded once per process."""
    conn = _DUCK.get("conn")
    if conn is None:
        import duckdb
        folder = os.path.join(ST, "tests", "fixtures", "optimizer", "tpc-ds", "")
        conn = duckdb.connect(config={"threads": 1})
        for table, columns in schema.items():
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM READ_CSV('{folder}{table}.csv.gz', delim='|', "
                         f"header=True, columns={columns})")
        _DUCK["conn"] = conn
    return conn


def test_comparison(sql, table, schema):
    """The test's own comparison for one query (subtestHelper): DuckDB's answer to the DuckDB text that transpile()
    makes now, against the executor's rows, with assert_frame_equal(check_dtype=False, check_index_type=False).
    Integers against floats and float sums within its tolerance pass there, not in result_hash."""
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from sqlglot import transpile
    a = duck_tables(schema).execute(transpile(sql, write="duckdb")[0]).fetchdf()
    b = pd.DataFrame(((np.nan if c is None else c for c in r) for r in table.rows), columns=table.columns)
    try:
        assert_frame_equal(a, b, check_dtype=False, check_index_type=False)
    except AssertionError:
        return False
    return True


def duck_answers_differ(dd, schema):
    """The test compares the executor's rows with DuckDB's answer to the DuckDB text of the query. Where the patched
    code makes another text, DuckDB answers both texts here, over the same views the test makes: the same frame
    (columns, dtypes, values, index) means the same comparison as at the pinned commit. None when every pair agrees,
    else why not. One thread, and the rows the test's views read loaded once (~0.8 s CPU; ~13 ms a query after)."""
    if len(dd) > DUCK_MAX_TEXTS:
        return f"{len(dd)} texts"
    import duckdb
    folder = os.path.join(ST, "tests", "fixtures", "optimizer", "tpc-ds", "")
    conn = duckdb.connect(config={"threads": 1})
    try:
        for table, columns in schema.items():
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM READ_CSV('{folder}{table}.csv.gz', delim='|', "
                         f"header=True, columns={columns})")
        for q, text in dd:
            a = conn.execute(q["duck"]).fetchdf()
            try:
                b = conn.execute(text).fetchdf()
            except Exception as exc:  # noqa: BLE001  (the test fails on it)
                return f"{q['i']}: {type(exc).__name__}"
            if list(a.columns) != list(b.columns) or not a.dtypes.equals(b.dtypes) or not a.index.equals(b.index)                     or not a.equals(b):
                return f"{q['i']}: another answer"
    finally:
        conn.close()
    return None


def query_check(job, tmp):
    """The cheap stages of test_execute_tpcds on the tree at tmp (imported here), against the pinned outputs. With
    tmp None: on the sqlglot this process already has (the pinned tree with the commit's code put in)."""
    if True:
        if tmp is not None:
            sys.path.insert(0, tmp)
        import sqlglot
        if not os.path.abspath(sqlglot.__file__).startswith(tmp if tmp is not None else os.path.join(ST, "")):
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
        phase_limit(max(8.0, 3 * (job["est"] + job.get("xbudget", 0.0)) + 2))
        for q in job["pn"]:
            pickle.loads(pickle.dumps(parse_one(q["sql"])))
        dd = []
        for q in job["d"]:  # cheap: before any query runs
            text = transpile(q["sql"], write="duckdb")[0]
            if text != q["duck"]:
                dd.append((q, text))
        if dd:
            why = duck_answers_differ(dd, job["schema"])
            if why:
                print("duckdb sql differs", why)
                return 1
        diffs = []
        static = {q["i"] for q in job.get("xq", [])}
        extra = 0.0
        ran = set()
        cache = {}
        shared_schema = None

        def same_result(q):
            # mp_execute in the test's child: the tables the query names, execute() with the shared schema, the
            # result pickled back to the parent, which compares it with duckdb's. Same result as at the pinned
            # commit, where the test passes -> same comparison.
            nonlocal shared_schema
            if shared_schema is None:
                shared_schema = pickle.loads(pickle.dumps(job["schema"]))
            expression = pickle.loads(pickle.dumps(parse_one(q["sql"])))
            tables = {}
            for t in find_tables(expression):
                name = t.name
                if name not in cache:
                    with open(os.path.join(TP, name + ".pkl"), "rb") as fh:
                        cache[name] = pickle.load(fh)
                tables[name] = cache[name]
            table = pickle.loads(pickle.dumps(execute(expression, schema=shared_schema, tables=tables)))
            if result_hash(table) == q["res"]:
                print("x", q["i"], "h")
                return True
            same = test_comparison(q["sql"], table, job["schema"])
            print("x", q["i"], "f" if same else "n")
            return same

        for q in job["q1"]:
            e = pickle.loads(pickle.dumps(parse_one(q["sql"])))
            schema = sx.ensure_schema(copy.deepcopy(job["schema"]), dialect=None)
            o = sx.optimize(e, schema, leave_tables_isolated=True, dialect=None)
            o.sql(pretty=True)
            if fingerprint(o) != q["fo"]:
                print("o", q["i"])
                diffs.append(q)  # the executor gets another tree: execute the query and compare its result
                if q["i"] in static:
                    continue
                extra += q["xcpu"]
                if extra > job.get("xbudget", 0.0):  # it cannot be dropped any more: stop spending CPU
                    print("optimize differs", [x["i"] for x in diffs], f"executing them would cost {extra:.1f} s")
                    return 1
                if q["xcpu"] > CHECK_X_INLINE_S:
                    continue  # runs once every tree is known and the whole set fits the budget
                # A cheap one right away: where the test fails, a differing result usually shows early.
                if not tables_ready():
                    print("no tables")
                    return 1
                if not same_result(q):
                    print("result differs", q["i"])
                    return 1
                ran.add(q["i"])
        _log(f"check: {len(diffs)} optimized trees differ, executed {len(ran)} ({extra:.1f} s estimated)")
        for q in job.get("px", []):
            # A query the test only parses (parent) and pickles to a child that returns None for it: it passes when
            # both work. The tree may not be deeper than the pinned query's (pickling recurses).
            a = parse_one(q["sql"])
            pickle.loads(pickle.dumps(a))
            if tree_depth(a) > tree_depth(parse_one(q["old"])) + 3:
                print("parsed query deeper", q["i"])
                return 1
        run = {q["i"]: q for q in job.get("xq", []) + diffs if q["i"] not in ran}
        if run:
            if not tables_ready():
                print("no tables")
                return 1
            phase_limit(max(5.0, 3 * sum(q["xcpu"] for q in run.values()) + 2))
            for q in sorted(run.values(), key=lambda q: q["xcpu"]):
                if not same_result(q):
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


def skey(k):
    """A text for a dict key or set member that is the same in every process, and the same for keys that are equal
    as dict keys (1, 1.0 and True; a str enum member and its text). Keys of other kinds fall into one text per type:
    unequal keys may then share a text, which only widens what reads it."""
    import enum
    if isinstance(k, str):
        return "s:" + str.__str__(k)
    if k is None:
        return "N"
    if isinstance(k, (bool, int)):
        return "i:%d" % int(k)
    if isinstance(k, float):
        return "i:%d" % int(k) if k.is_integer() else "f:" + repr(k)
    if isinstance(k, tuple):
        return "(" + ",".join(skey(x) for x in k) + ")"
    if isinstance(k, frozenset):
        return "{" + ",".join(sorted(skey(x) for x in k)) + "}"
    if isinstance(k, enum.Enum):
        t = type(k)
        return f"E:{t.__module__}.{t.__qualname__}.{k.name}"
    if isinstance(k, type):
        return f"C:{k.__module__}.{k.__qualname__}"
    t = type(k)
    return f"T:{getattr(t, '__module__', '?')}.{getattr(t, '__qualname__', '?')}"


TABLE_TYPES = (dict, set, frozenset)


def snap_state(root, bag_keys=(), want_ents=None, roots=None, errs=None):
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
    if errs is None:  # else the caller imported everything (the warm key check)
        errs = {}
        for name in sorted(x.name for x in pkgutil.walk_packages(sqlglot.__path__, "sqlglot.", onerror=lambda n: None)):
            try:
                importlib.import_module(name)
            except BaseException as exc:  # noqa: BLE001
                errs[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
    gc.collect()
    stats = {"opaque": 0, "deep": 0}
    cur, okeys = [None], set()  # the key being printed; keys whose value holds an object known only by its repr
    addr = re.compile(r" at 0x[0-9a-fA-F]+")
    memo = {}

    names = {}  # id(class) -> (class, its name text, its reference text): the same text every time it is asked

    def qn(t):
        hit = names.get(id(t))
        if hit is not None and hit[0] is t:
            return hit[1]
        q = f"{getattr(t, '__module__', '?')}.{getattr(t, '__qualname__', getattr(t, '__name__', '?'))}"
        if isinstance(t, type):
            names[id(t)] = (t, q, None)
        return q

    def cls_ref(o):
        hit = names.get(id(o))
        if hit is not None and hit[0] is o and hit[2] is not None:
            return hit[2]
        ref = f"T:{qn(o)}<{','.join(qn(b) for b in o.__bases__)}>"
        names[id(o)] = (o, qn(o), ref)
        return ref

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
            hit = memo.get((id(o), bag))
            if hit is not None and hit[0] is o:
                return hit[1]
            out = f"E:{qn(t)}.{o.name}=" + fp(o._value_, d + 1, stack, bag)
            memo[(id(o), bag)] = (o, out)
            return out
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
                    okeys.add(cur[0])
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

    keys, bags, done, ents = {}, {}, set(), {}
    bag_keys = set(bag_keys)

    def put(key, value, t, mod, qual):
        cur[0] = key
        keys[key] = [h(fp(value, 0, frozenset(), False)), t, mod, qual]
        if key in bag_keys:
            bags[key] = h(fp(value, 0, frozenset(), True))
        if type(value) in TABLE_TYPES and want_ents is not None and \
                (want_ents is True or want_ents(key, keys[key][0], bags.get(key))):
            # Entry by entry, so that a diff can name the keys that changed (values order-free where the holder's
            # value differs from run to run).
            per, order = {}, []
            for k in value:
                v = value[k] if type(value) is dict else True
                try:
                    sk = skey(k)
                except Exception:  # noqa: BLE001
                    sk = "?"
                order.append(sk)
                per.setdefault(sk, []).append(fp(v, 0, frozenset(), key in bag_keys))
            e = {sk: h("|".join(sorted(x))) for sk, x in per.items()}
            e["#n"] = str(len(value))
            if type(value) is dict and key not in bag_keys:
                e["#order"] = h("\x00".join(order))
            ents[key] = e

    def do_class(c, path, mod, qual, inroot=True):
        if id(c) in done:
            return
        done.add(id(c))
        if inroot:
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
                if inroot:
                    keys[f"{path}.{k}"] = [h(cls_ref(v)), "T", cmod, cqual]
                do_class(v, f"{path}.{k}", mod, f"{qual}.{k}", inroot)
            elif inroot or f"{path}.{k}" == DIALECT_TABLE:
                put(f"{path}.{k}", v, tag(v), cmod, cqual)

    for n, mo in sorted((n, mo) for n, mo in sys.modules.items() if n == "sqlglot" or n.startswith("sqlglot.")):
        dct = vars(mo)
        inroot = roots is None or n in roots
        if inroot:
            keys[n + "::order"] = [h(",".join(k for k in dct if k not in SNAP_SKIP_MOD)), "S", n, ""]
        for k, v in dct.items():
            if k in SNAP_SKIP_MOD:
                continue
            if isinstance(v, type) and v.__module__ == n and v.__qualname__ == k:
                if inroot:
                    keys[f"{n}:{k}"] = [h(cls_ref(v)), "GT", n, ""]
                do_class(v, f"{n}.{k}", n, k, inroot)
            elif not inroot:
                continue
            elif isinstance(v, types.ModuleType):
                put(f"{n}:{k}", v, "GM", n, "")
            else:
                put(f"{n}:{k}", v, {"T": "GT", "F": "GF"}.get(tag(v), "G"), n, "")
    for n, e in errs.items():
        keys["IMPORTERROR:" + n] = [h(e), "E", n, ""]
    stats["modules"] = len([n for n in sys.modules if n == "sqlglot" or n.startswith("sqlglot.")])
    stats["errors"] = len(errs)
    stats["ents"] = ents
    stats["okeys"] = sorted(okeys)
    return keys, bags, stats


def graph_main(argv):
    """--graph ROOT OUT: which sqlglot modules each sqlglot module imports while it loads (import statements,
    importlib, parents of what it imports), and which modules change objects outside their own namespace while they
    load (an augmented assignment to an item or attribute, a mutating call on a name the same body did not make, an
    item or attribute assignment, setattr): the warm key check re-imports only a changed module and its importers,
    which is exact only when nothing loading changes what it did not make."""
    import builtins
    import importlib
    import importlib.util
    import pkgutil
    import unittest  # noqa: F401
    root, out = argv
    sys.setrecursionlimit(20000)
    sys.path.insert(0, root)
    edges = {}
    real_import, real_im = builtins.__import__, importlib.import_module
    # an edge goes from the module whose top-level code is running (the innermost module being loaded) to what it
    # imports, also through a call made while it loads (a lazy package attribute, Dialect.get_or_raise)
    from importlib import _bootstrap_external as be
    loading = []
    real_exec = be._LoaderBasics.exec_module

    def exec_hook(self, module):
        loading.append(module.__name__)
        try:
            return real_exec(self, module)
        finally:
            loading.pop()

    def note(dst):
        src = loading[-1] if loading else None
        if src and dst and src != dst and src.partition(".")[0] == "sqlglot" and dst.partition(".")[0] == "sqlglot":
            parts = dst.split(".")
            for i in range(1, len(parts) + 1):
                d = ".".join(parts[:i])
                if d != src:
                    edges.setdefault(src, set()).add(d)

    def note_obj(o):
        if isinstance(o, types.ModuleType):
            note(o.__name__)
        else:
            m = getattr(o, "__module__", None)
            if isinstance(m, str):
                note(m)

    def hook(name, globals=None, locals=None, fromlist=(), level=0):
        mod = real_import(name, globals, locals, fromlist, level)
        a = name
        if level:
            try:
                a = importlib.util.resolve_name("." * level + name, (globals or {}).get("__package__") or "")
            except (ImportError, ValueError):
                a = None
        if a:
            note(a)
            for f in fromlist or ():
                if f != "*":
                    note_obj(getattr(sys.modules.get(a), f, None))
        return mod

    def hook_im(name, package=None):
        mod = real_im(name, package)
        note(mod.__name__)
        return mod
    import types
    be._LoaderBasics.exec_module = exec_hook
    builtins.__import__, importlib.import_module = hook, hook_im
    import sqlglot
    names = sorted(x.name for x in pkgutil.walk_packages(sqlglot.__path__, "sqlglot.", onerror=lambda n: None))
    for name in names:
        try:
            real_im(name)
        except BaseException:  # noqa: BLE001
            pass
    builtins.__import__, importlib.import_module = real_import, real_im
    be._LoaderBasics.exec_module = real_exec
    mutating = {"pop", "popitem", "update", "add", "clear", "remove", "discard", "setdefault", "extend", "append",
                "insert", "__setitem__", "__delitem__", "sort", "reverse", "difference_update", "set",
                "intersection_update", "symmetric_difference_update", "__ior__", "__iadd__", "__setattr__"}
    fresh_values = (ast.Dict, ast.Set, ast.List, ast.Tuple, ast.DictComp, ast.SetComp, ast.ListComp,
                    ast.GeneratorExp, ast.Call, ast.BinOp, ast.Constant, ast.JoinedStr, ast.Lambda, ast.BoolOp,
                    ast.Compare, ast.UnaryOp, ast.IfExp)

    def calls(node):
        todo = [node]
        while todo:
            n = todo.pop()
            if isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(n, ast.Call):
                yield n
            todo.extend(ast.iter_child_nodes(n))
    unsafe = []
    for name in ["sqlglot"] + names:
        mo = sys.modules.get(name)
        path = getattr(mo, "__file__", None) if mo else None
        if not path:
            continue
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (OSError, SyntaxError, ValueError):
            unsafe.append(name)
            continue
        bad = []

        def owned(t, own):  # a store into an object this body made itself
            return isinstance(t.value, ast.Name) and t.value.id in own

        def check(node, own):
            for c in calls(node):
                f = c.func
                if isinstance(f, ast.Name) and f.id in ("setattr", "delattr"):
                    bad.append(c.lineno)
                elif isinstance(f, ast.Attribute) and f.attr in mutating and not \
                        (isinstance(f.value, ast.Name) and f.value.id in own):
                    bad.append(c.lineno)

        def body(stmts, own):
            for st in stmts:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    for d in st.decorator_list:
                        check(d, own)
                    if isinstance(st, ast.ClassDef):
                        for b in st.bases + [k.value for k in st.keywords]:
                            check(b, own)
                        body(st.body, set())
                    own.add(st.name)
                    continue
                if isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    if st.value is not None:
                        check(st.value, own)
                    targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                    if isinstance(st, ast.AugAssign):
                        if not (isinstance(st.target, ast.Name) and st.target.id in own):
                            bad.append(st.lineno)
                        continue
                    for t in targets:
                        for n in ast.walk(t):
                            if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, ast.Store):
                                if not owned(n, own):
                                    bad.append(st.lineno)
                                check(n.value, own)
                                if isinstance(n, ast.Subscript):
                                    check(n.slice, own)
                            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                                if isinstance(st.value, fresh_values) and isinstance(t, ast.Name):
                                    own.add(n.id)
                                else:
                                    own.discard(n.id)
                    continue
                if isinstance(st, ast.Delete):
                    for t in st.targets:
                        if isinstance(t, (ast.Attribute, ast.Subscript)) and not owned(t, own):
                            bad.append(st.lineno)
                    continue
                if isinstance(st, (ast.If, ast.While)):
                    check(st.test, own)
                elif isinstance(st, (ast.For, ast.AsyncFor)):
                    check(st.iter, own)
                    for n in ast.walk(st.target):
                        if isinstance(n, (ast.Attribute, ast.Subscript)):
                            bad.append(st.lineno)
                        elif isinstance(n, ast.Name):
                            own.discard(n.id)
                elif isinstance(st, (ast.With, ast.AsyncWith)):
                    for it in st.items:
                        check(it.context_expr, own)
                elif isinstance(st, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                    for hd in st.handlers:
                        if hd.type is not None:
                            check(hd.type, own)
                elif isinstance(st, ast.Match):
                    bad.append(st.lineno)
                elif not isinstance(st, (ast.Import, ast.ImportFrom, ast.Pass, ast.Global, ast.Nonlocal,
                                         ast.Break, ast.Continue)):
                    check(st, own)
                for f in ("body", "orelse", "finalbody"):
                    if getattr(st, f, None):
                        body(getattr(st, f), own)
                for hd in getattr(st, "handlers", []):
                    body(hd.body, own)
        body(tree.body, set())
        if bad:
            unsafe.append(name)
    with open(out, "w") as fh:
        json.dump({"edges": {k: sorted(v) for k, v in edges.items()}, "unsafe": sorted(set(unsafe)),
                   "modules": sorted(n for n in sys.modules if n == "sqlglot" or n.startswith("sqlglot."))}, fh)
    return 0


def snap_env():
    """The snapshot child's environment: a fixed hash seed (sets of strings iterate the same in both trees)."""
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": "/tmp", "PYTHONHASHSEED": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "SKIP_INTEGRATION": "0"}
    if SET_PLANS:
        env["BR_SET"] = "1"
    if os.environ.get("BR_SNAP_ALLOC", "1") == "1":  # the allocator the image sets (build and run alike)
        env.update((k, v) for k, v in os.environ.items() if k == "PYTHONMALLOC" or k.startswith("MIMALLOC_"))
    return env


def snap_run(root, out, bag_keys=(), pad=0, ents=False):
    """One full snapshot of root in a fresh process, written to out. pad shifts the process's memory layout (the
    order of sets of classes follows their addresses)."""
    with open(out + ".bag", "w") as fh:
        json.dump(sorted(bag_keys), fh)
    env = dict(snap_env(), BR_SNAP_PAD=str(pad))
    subprocess.run([sys.executable, "-s", "-P", os.path.abspath(__file__), "--snap", root, out, out + ".bag"]
                   + (["ents"] if ents else []), env=env, check=True, cwd="/tmp", stdout=subprocess.DEVNULL)
    with open(out) as fh:
        return json.load(fh)


def build_state_base(root, dest):
    """Build time: the pinned tree's import-time state, eight times from copies at paths of different lengths and
    with different memory layouts, to find values whose order or content differs from run to run; those are
    compared as bags (order-free), twice more to make sure the bags agree."""
    import random
    tmp = tempfile.mkdtemp(prefix="sd")
    rng = random.Random(20261122)

    def run(i, bag_keys=(), ents=False):
        copy_root = os.path.join(tmp, "t" * (1 + rng.randrange(40)) + str(i))
        shutil.copytree(os.path.join(root, "sqlglot"), os.path.join(copy_root, "sqlglot"))
        try:
            return snap_run(copy_root, os.path.join(tmp, f"s{i}.json"), bag_keys, pad=rng.randrange(0, 200000),
                            ents=ents)
        finally:
            shutil.rmtree(copy_root, ignore_errors=True)
    try:
        runs = [run(i) for i in range(8)]
        base = runs[0]["keys"]
        unstable = sorted({k for r in runs[1:] for k in set(base) | set(r["keys"]) if r["keys"].get(k) != base.get(k)})
        b1 = run(8, unstable, ents=True)
        b2 = run(9, unstable, ents=True)
        never = sorted(k for k in unstable if b1["bags"].get(k) != b2["bags"].get(k))
        if any(r["stats"]["deep"] for r in runs):
            raise RuntimeError(f"pinned state not comparable: {runs[0]['stats']}")
        ents = b1["stats"].pop("ents", {})
        # entries that differ between the two bag runs can never be compared entry by entry
        e2 = b2["stats"].pop("ents", {})
        ent_never = sorted(k for k in ents if ents[k] != e2.get(k))
        for r in runs:
            r["stats"].pop("ents", None)
        gout = os.path.join(tmp, "graph.json")
        subprocess.run([sys.executable, "-s", "-P", os.path.abspath(__file__), "--graph", root, gout],
                       env=snap_env(), check=True, cwd="/tmp", stdout=subprocess.DEVNULL)
        with open(gout) as fh:
            graph = json.load(fh)
        okeys = runs[0]["stats"].pop("okeys", [])
        for r in runs:
            r["stats"].pop("okeys", None)
        sd = {"keys": base, "unstable": unstable, "bags": b1["bags"], "never": never, "stats": runs[0]["stats"],
              "ents": ents, "ent_never": ent_never, "okeys": okeys, "graph": graph}
        with open(dest + ".graph.json", "w") as fh:  # the main process's read: which changes the warm check takes
            json.dump(graph, fh)
        with open(dest, "w") as fh:
            json.dump(sd, fh)
        with open(dest + ".pkl", "wb") as fh:  # the key check's fast read of the same base
            pickle.dump(sd, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print("state base:", len(base), "keys,", len(unstable), "unstable,", len(never), "never comparable,",
              len(ents), "tables,", len(ent_never), "tables not comparable by entry,", runs[0]["stats"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def snap_main(argv):
    """--snap ROOT OUT BAGFILE [ents]: write one snapshot (with every table's entry prints when "ents")."""
    root, out, bagfile = argv[:3]
    with open(bagfile) as fh:
        bag_keys = json.load(fh)
    pad = int(os.environ.get("BR_SNAP_PAD", "0") or 0)
    ballast = [object() for _ in range(pad)], bytearray(pad * 7)  # noqa: F841  (shifts addresses; kept alive)
    keys, bags, stats = snap_state(root, bag_keys, True if argv[3:4] == ["ents"] else None)
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


def key_snap_main(argv):
    """--key-snap ROOT OUT: snapshot ROOT and write its differences from the base (key_diff) as JSON."""
    root, out = argv
    with open(SD + ".pkl", "rb") as fh:
        base = pickle.load(fh)
    unstable = set(base["unstable"])

    def want(key, whole, bag):
        if key in unstable:
            return bag != base["bags"].get(key)
        rec = base["keys"].get(key)
        return rec is None or rec[0] != whole
    try:
        keys, bags, stats = snap_state(root, base["unstable"], want)
    except Exception as exc:  # noqa: BLE001
        r = {"why": f"snapshot failed: {exc!r}"}
    else:
        r = key_diff(base, {"keys": keys, "bags": bags, "stats": stats})
        if "why" not in r:
            r["shadow"] = shadows(r)
        if "why" not in r and SET_PLANS:
            r["values"] = diff_values(r)
    with open(out, "w") as fh:
        json.dump(r, fh)
    os._exit(0)  # no interpreter teardown


def key_diff(base, new, scope=None):
    """{"diffs": [[key, module, gone, base record, new record]], "by_key": {key: changed entry keys, or None for a
    holder not compared entry by entry}} ("#n": the size changed, "#order": the keys or their order), or
    {"why": reason} when the trees cannot be compared."""
    nents = new["stats"].pop("ents", {})
    nokeys = set(new["stats"].pop("okeys", ()))
    if scope is None:
        if new["stats"].get("deep") or new["stats"].get("opaque") != base["stats"].get("opaque"):
            return {"why": f"snapshot not comparable: {new['stats']} vs {base['stats']}"}
        base_keys = base["keys"]
    else:
        base_keys = {k: base["keys"][k] for k in scope if k in base["keys"]}
        if new["stats"].get("deep") or nokeys != set(base.get("okeys", ())) & set(base_keys) - {None}:
            return {"why": f"snapshot not comparable: {new['stats']}, opaque {sorted(nokeys)[:5]}"}
    unstable, never, ent_never = set(base["unstable"]), set(base["never"]), set(base.get("ent_never", ()))
    diffs, by_key = [], {}
    for k in sorted(set(base_keys) | set(new["keys"])):
        a, b = base_keys.get(k), new["keys"].get(k)
        if k in never:
            return {"why": f"never comparable: {k}"}
        if k in unstable:
            changed = a is None or b is None or base["bags"].get(k) != new["bags"].get(k) or a[1:] != b[1:]
        else:
            changed = a != b
        if not changed:
            continue
        diffs.append([k, (b or a)[2], b is None, a, b])
        ea, eb = base["ents"].get(k), nents.get(k)
        if ea is None or eb is None or k in ent_never:
            by_key[k] = None
            continue
        ch = {sk for sk in set(ea) | set(eb) if ea.get(sk) != eb.get(sk)}
        if set(ea) - {"#n", "#order"} != set(eb) - {"#n", "#order"}:
            ch.add("#order")
        by_key[k] = sorted(ch)
    return {"diffs": diffs, "by_key": by_key}


WARM_MAX_MODULES = 12  # the warm check re-imports at most this many modules (a changed module and its importers)
KEY_MIN_WARM_S = float(os.environ.get("BR_KEY_MIN_WARM", 2.5))  # with a warm check (~0.2 s), commits whose chosen tests cost this much are worth one


def mod_of(path):
    m = path[:-3].replace("/", ".")
    return m[: -len(".__init__")] if m.endswith(".__init__") else m


def warm_group(graph, files):
    """The modules a warm check re-imports for these changed files (each changed module and every module that
    imports one while loading), or None when the fresh snapshot must be used."""
    rev = graph.get("_rev")
    if rev is None:
        rev = graph["_rev"] = {}
        for a, ds in graph["edges"].items():
            for d in ds:
                rev.setdefault(d, set()).add(a)
    group = {mod_of(p) for p in files}
    if not group <= set(graph["modules"]):
        return None
    todo = list(group)
    while todo:
        for a in rev.get(todo.pop(), ()):
            if a not in group:
                group.add(a)
                todo.append(a)
    if len(group) > WARM_MAX_MODULES or group & set(graph["unsafe"]) or "sqlglot" in group:
        return None
    return group


class _TextLoader:
    """Loads one module from a text in memory, compiled as a .pyc would be."""

    def __init__(self, path, text):
        self.path, self.text = path, text

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        import marshal
        code = compile(self.text.encode("utf-8", "surrogateescape"), self.path, "exec", dont_inherit=True)
        exec(marshal.loads(marshal.dumps(code)), module.__dict__)


class _TextFinder:
    """First on sys.meta_path in a warm check's child: the changed modules come from their patched text."""

    def __init__(self, root, files):
        self.texts = {mod_of(p): (os.path.join(root, p), p.endswith("/__init__.py"), t) for p, t in files.items()}

    def find_spec(self, name, path=None, target=None):
        import importlib.util
        hit = self.texts.get(name)
        if hit is None:
            return None
        where, is_pkg, text = hit
        return importlib.util.spec_from_file_location(
            name, where, loader=_TextLoader(where, text),
            submodule_search_locations=[os.path.dirname(where)] if is_pkg else None)


def key_server_main(argv):
    """--key-server ROOT: import the pinned tree once, then for each job line on stdin ({"files", "mods", "out"})
    fork a child that re-imports the job's modules from the patched texts, snapshots only them, writes the
    key_diff answer to "out", and prints a line when it is done."""
    import importlib
    import pkgutil
    import unittest  # noqa: F401
    root = argv[0]
    sys.setrecursionlimit(20000)
    sys.path.insert(0, root)
    import sqlglot
    errs = {}
    for name in sorted(x.name for x in pkgutil.walk_packages(sqlglot.__path__, "sqlglot.", onerror=lambda n: None)):
        try:
            importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001
            errs[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
    with open(SD + ".pkl", "rb") as fh:
        base = pickle.load(fh)
    mods = sorted(base["graph"]["modules"], key=len, reverse=True)

    def root_of(k):
        for m in mods:
            if k.startswith(m) and k[len(m):len(m) + 1] in (".", ":"):
                return m
        return None
    by_root = {}
    for k in base["keys"]:
        by_root.setdefault(k[len("IMPORTERROR:"):] if k.startswith("IMPORTERROR:") else root_of(k), []).append(k)
    unstable = set(base["unstable"])

    def want(key, whole, bag):
        if key in unstable:
            return bag != base["bags"].get(key)
        rec = base["keys"].get(key)
        return rec is None or rec[0] != whole
    import gc
    make_immortal()
    gc.freeze()
    for line in sys.stdin:
        job = json.loads(line)
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                group = set(job["mods"])
                for m in sorted(group, key=len, reverse=True):
                    old = sys.modules.pop(m, None)
                    parent, _, leaf = m.rpartition(".")
                    if old is not None and parent in sys.modules and getattr(sys.modules[parent], leaf, None) is old:
                        delattr(sys.modules[parent], leaf)
                # a package with a lazy __getattr__ keeps what it handed out in its globals: drop the re-made ones
                for n, mo in list(sys.modules.items()):
                    if (n == "sqlglot" or n.startswith("sqlglot.")) and n not in group and "__getattr__" in vars(mo):
                        for a, v in list(vars(mo).items()):
                            if not isinstance(v, types.ModuleType) and getattr(v, "__module__", None) in group:
                                delattr(mo, a)
                sys.meta_path.insert(0, _TextFinder(root, job["files"]))
                new_errs = {}
                for m in sorted(group):
                    try:
                        importlib.import_module(m)
                    except BaseException as exc:  # noqa: BLE001
                        new_errs[m] = f"{type(exc).__name__}: {str(exc)[:200]}"
                # a re-made package does not get back its submodules that stayed loaded (the import system sets that
                # attribute only when it loads the submodule): set them, as a fresh import of the tree has them
                for n, mo in list(sys.modules.items()):
                    parent, _, leaf = n.rpartition(".")
                    pm = sys.modules.get(parent) if parent in group and n not in group else None
                    if pm is not None and mo is not None and leaf not in vars(pm):
                        setattr(pm, leaf, mo)
                keys, bags, stats = snap_state(root, base["unstable"], want, roots=group, errs=new_errs)
                scope = {k for m in group for k in by_root.get(m, ())} | {DIALECT_TABLE}
                r = key_diff(base, {"keys": keys, "bags": bags, "stats": stats}, scope)
                if "why" not in r:
                    r["shadow"] = shadows(r)
                if "why" not in r and SET_PLANS:
                    r["values"] = diff_values(r)
                with open(job["out"], "w") as fh:
                    json.dump(r, fh)
                code = 0
            except BaseException as exc:  # noqa: BLE001
                try:
                    with open(job["out"], "w") as fh:
                        json.dump({"why": f"warm check failed: {exc!r}"}, fh)
                except OSError:
                    pass
            finally:
                os._exit(code)
        os.waitpid(pid, 0)
        print("done", flush=True)
    os._exit(0)  # no interpreter teardown


def key_snap(files):
    """The key check of a patched tree ({path: text}), in one fresh process; its answer (dict)."""
    tmp = patched_tree(files, pyc=True)
    try:
        out = os.path.join(tmp, "keys.json")
        subprocess.run([sys.executable, "-s", "-P", os.path.abspath(__file__), "--key-snap", tmp, out],
                       env=snap_env(), check=True, cwd="/tmp", stdout=subprocess.DEVNULL, timeout=60)
        with open(out) as fh:
            return json.load(fh)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


KEY_BUDGET_S = float(os.environ.get("BR_KEY_BUDGET", 60))  # no key check starts after this many seconds
KEY_MIN_S = 6.0  # a key check runs only where the chosen tests cost this many simulated seconds (below, the runs
# settle a commit for less CPU than the check)
LAZY_PKGS = ("sqlglot.dialects", "sqlglot.optimizer")  # packages whose __getattr__ imports and caches names
DIALECT_TABLE = "sqlglot.dialects.dialect._Dialect._classes"  # dialect name -> class; a lookup of a name not there imports


def values_only(old_text, new_text):
    """True when a change edits only the values of NAME = ... and NAME: T = ... statements in module and class bodies
    (also inside module-level if / try / with blocks), None when the code is the same (comments, blank lines),
    False otherwise."""
    try:
        a, b = ast.parse(old_text), ast.parse(new_text)
    except (SyntaxError, ValueError):
        return False
    if "\r" not in old_text and "\r" not in new_text:
        return values_only_by_statement(a, b, old_text.split("\n"), new_text.split("\n"))
    if ast.dump(a) == ast.dump(b):
        return None

    def blank(body):
        for st in body:
            if isinstance(st, ast.Assign) and all(isinstance(x, ast.Name) for x in st.targets):
                st.value = ast.Constant(None)
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
                st.value = ast.Constant(None)
            elif isinstance(st, ast.ClassDef):
                blank(st.body)
            elif isinstance(st, ast.If):
                blank(st.body)
                blank(st.orelse)
            elif isinstance(st, (ast.Try, ast.With)):
                for f in ("body", "orelse", "finalbody"):
                    blank(getattr(st, f, []))
                for hd in getattr(st, "handlers", []):
                    blank(hd.body)
    blank(a.body)
    blank(b.body)
    return ast.dump(a) == ast.dump(b)


def values_only_by_statement(a, b, la, lb):
    """values_only on parsed trees, comparing only the statements whose source text differs (a statement with the
    same text at the same columns parses to the same tree), recursing into class bodies. Same answers as dumping
    both whole trees."""
    def seg(lines, st):
        # a decorated def / class starts at its first decorator, above its lineno
        first = min([st.lineno] + [d.lineno for d in getattr(st, "decorator_list", ())])
        return st.col_offset, st.end_col_offset, lines[first - 1:st.end_lineno]

    def kind(st):
        return type(st), getattr(st, "name", None)

    def differing(x, y):
        if len(x) != len(y):
            return None
        d = [(s, t) for s, t in zip(x, y) if seg(la, s) != seg(lb, t)]
        return d if all(kind(s) == kind(t) for s, t in d) else None

    def constant(st):  # NAME = ... (not a dunder or _sunder_ name)
        if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
            return False
        return not (st.targets[0].id.startswith("_") and st.targets[0].id.endswith("_"))

    def aligned(x, y, in_class):
        """Bodies that differ by defs moved within the body (the same def elsewhere) and, in an undecorated class, by
        constants gained or lost: the pairs of statements whose text differs, or None."""
        import difflib
        kx = [(s.col_offset, s.end_col_offset, tuple(seg(la, s)[2])) for s in x]
        ky = [(t.col_offset, t.end_col_offset, tuple(seg(lb, t)[2])) for t in y]
        pairs, gone, new = [], [], []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, kx, ky, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            if op == "replace" and i2 - i1 == j2 - j1 and all(kind(s) == kind(t) for s, t in zip(x[i1:i2], y[j1:j2])):
                pairs += zip(x[i1:i2], y[j1:j2])
            else:
                gone += x[i1:i2]
                new += y[j1:j2]
        defs = (ast.FunctionDef, ast.AsyncFunctionDef)
        if sorted(ast.dump(s) for s in gone if isinstance(s, defs)) != sorted(
                ast.dump(t) for t in new if isinstance(t, defs)):
            return None
        if any(not isinstance(st, defs) and not (in_class and constant(st)) for st in gone + new):
            return None
        return pairs
    top = differing(a.body, b.body)
    moved = top is None
    if moved:
        top = aligned(a.body, b.body, False)
        if top is None:
            return False
    if not moved and all(ast.dump(s) == ast.dump(t) for s, t in top):
        return None

    def blank1(st):
        if isinstance(st, ast.Assign) and all(isinstance(x, ast.Name) for x in st.targets):
            st.value = ast.Constant(None)
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
            st.value = ast.Constant(None)
        elif isinstance(st, ast.If):
            for x in st.body + st.orelse:
                blank1(x)
        elif isinstance(st, (ast.Try, ast.With)):
            for f in ("body", "orelse", "finalbody"):
                for x in getattr(st, f, []):
                    blank1(x)
            for hd in getattr(st, "handlers", []):
                for x in hd.body:
                    blank1(x)
        elif isinstance(st, ast.ClassDef):
            for x in st.body:
                blank1(x)

    def same(s, t):
        if isinstance(s, ast.ClassDef) and isinstance(t, ast.ClassDef):
            bs, bt = s.body, t.body
            s.body, t.body = [], []
            head = ast.dump(s) == ast.dump(t)
            s.body, t.body = bs, bt
            if not head:
                return False
            inner = differing(bs, bt)
            if inner is None:
                inner = aligned(bs, bt, not s.decorator_list)
            return inner is not None and all(same(x, y) for x, y in inner)
        blank1(s)
        blank1(t)
        return ast.dump(s) == ast.dump(t)
    return all(same(s, t) for s, t in top)


_WORD = re.compile(r"(\W+)")


def renames(text):
    """{new: old} identifier renames that explain every changed line of an in-place patch (lines replaced one for one,
    each differing from the old only in identifiers, each new name always replacing the same old one), or None."""
    pairs = {}
    for old, new, blocks in parse_patch(text):
        if not old or new != old:
            return None
        for removed, added, _ in blocks:
            if len(removed) != len(added):
                return None
            for (_, a), b in zip(removed, added):
                ta, tb = _WORD.split(a), _WORD.split(b)
                if len(ta) != len(tb):
                    return None
                for x, y in zip(ta, tb):
                    if x != y and not (x.isidentifier() and y.isidentifier() and pairs.setdefault(y, x) == x):
                        return None
    return pairs or None


def novel_renames(m, text):
    """renames(text) when every new name is unknown to the pinned tree (no sqlglot or test file has the word) and
    cannot be found by a name the code builds (`*_sql` dispatch, dunders, _sunder_ names), or None."""
    rn = renames(text)
    if not rn or set(rn) & set(rn.values()):
        return None
    for new in rn:
        if new.endswith("_sql") or (new.startswith("_") and new.endswith("_")):
            return None
    pat = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in sorted(rn)) + r")\b")
    for p in m.d.get("src", {}):
        s = m.source(p)
        if s is None or pat.search(s):
            return None
    return rn


def key_job(m, text):
    """{"files": {path: patched text}, "data": [paths]} for a change that edits values of assignments in sqlglot
    modules, besides comments there and .sql fixtures (whose readers stay), or None. With "renames" ({new: old})
    for a change that renames identifiers to new names (the key check prices the functions it touched)."""
    files, data, real, rn = {}, [], False, False
    for old, new, blocks in parse_patch(text):
        if not old or new != old:
            return None
        if old.startswith("sqlglot/") and old.endswith(".py"):
            if old not in m.d["files"]:
                return None
            src = m.source(old)
            nt = patched(src, blocks) if src is not None else None
            v = values_only(src, nt) if nt is not None else False
            if v is False and nt is not None:
                if rn is False:
                    rn = novel_renames(m, text)
                if rn and re.sub(r"\b(?:" + "|".join(re.escape(n) for n in sorted(rn)) + r")\b",
                                 lambda mt: rn[mt.group(0)], nt) == src:
                    v = True
            if v is False:
                return None
            real = real or v
            files[old] = nt
        elif old.startswith("tests/fixtures/") and old.endswith(".sql"):
            data.append(old)
        else:
            return None
    if not real:
        return None
    job = {"files": files, "data": data}
    if rn:
        job["renames"] = rn
    return job


# Set plans. When the key check of a commit finds that every import-time difference is a plain value (None, bool,
# number, str, bytes, and tuples, lists, dicts and sets of those) held by a sqlglot module global or by a class's own
# namespace, its child also hands back those values. A forked run child can then set them on the objects the runner
# parent imported instead of importing the patched tree again (a hot plan without code pairs).
# Off by default: v25 ran them and the board's CPU rose 0.8 s with the simulated time unchanged.
SET_PLANS = os.environ.get("BR_SET") == "1"
_SET_PLANS = {}  # commit -> [(module, class path, name, value)]
_MATE_FAILED = set()  # commits where a classmate of test_execute_tpcds failed in the run
_SEEN_FAIL = {}  # commit -> tests that failed in its run
_HOW = {}  # commit -> how its tests were decided (readout only): runner mode / "declined" / "doubt" / ...
_KEYED = {}  # commit -> "n" narrowed by the key check, "a" kept all, "b" a module no longer imports
_TPC = {}  # commit -> "k" the TPC-DS check kept the test, "d" dropped it, "f" its job was filtered out
_MATES_RAN = set()  # commits whose run held a classmate of test_execute_tpcds
_TPC_WHY = {}  # commit -> why the TPC-DS check kept the test
_CTX = {}  # readout only: the map, the patch texts, the start time
_TEST_SOURCES = []


def _plain(v, depth=0):
    if depth > 6:
        return False
    if v is None or type(v) in (bool, int, str, bytes) or (type(v) is float and v == v and abs(v) != float("inf")):
        return True
    if type(v) in (tuple, list, set):
        return all(_plain(x, depth + 1) for x in v)
    if type(v) is dict:
        return all(_plain(k, depth + 1) and _plain(x, depth + 1) for k, x in v.items())
    return False


def diff_values(r):
    """In a key-check child, after key_diff: {key: repr of the new value} for every changed key that holds a plain
    value as a module global or in a class's own namespace (reached as module.qualname), where the repr reads back
    equal; keys that cannot be handed back are left out."""
    out = {}
    for k, mod, gone, _, b in r.get("diffs", ()):
        if gone or b is None or b[1] not in ("V", "G"):
            continue
        mo = sys.modules.get(b[2])
        if mo is None:
            continue
        if b[1] == "G":
            if not k.startswith(b[2] + ":"):
                continue
            holder, name = vars(mo), k[len(b[2]) + 1:]
        else:
            head = f"{b[2]}.{b[3]}."
            name = k[len(head):]
            if not b[3] or not k.startswith(head) or "." in name:
                continue
            o = mo
            for q in b[3].split("."):
                o = getattr(o, q, None)
            if not isinstance(o, type) or o.__module__ != b[2] or o.__qualname__ != b[3]:
                continue
            holder = o.__dict__
        if name not in holder or not _plain(holder[name]):
            continue
        try:
            text = repr(holder[name])
            if len(text) > 200000 or ast.literal_eval(text) != holder[name]:
                continue
        except Exception:  # noqa: BLE001
            continue
        out[k] = text
    return out


def shadows(r):
    """In a key-check child, after key_diff: for each attribute a class's own namespace gained, [the key of the one
    it now hides (module.qualname.name of the first class after it in its MRO that defines the name) or None,
    whether both are the same plain value]."""
    out = {}
    for k, mod, gone, a, b in r.get("diffs", ()):
        if gone or a is not None or b is None or b[1] != "V" or not b[3]:
            continue
        head = f"{b[2]}.{b[3]}."
        name = k[len(head):] if k.startswith(head) else ""
        mo = sys.modules.get(b[2])
        if not name or "." in name or mo is None:
            continue
        o = mo
        for q in b[3].split("."):
            o = getattr(o, q, None)
        if not isinstance(o, type) or o.__module__ != b[2] or o.__qualname__ != b[3] or name not in o.__dict__:
            continue
        v = o.__dict__[name]
        out[k] = [None, False]
        for base in o.__mro__[1:]:
            if name in base.__dict__:
                w = base.__dict__[name]
                try:
                    same = _plain(v) and _plain(w) and type(v) is type(w) and v == w and repr(v) == repr(w)
                except Exception:  # noqa: BLE001
                    same = False
                out[k] = [f"{base.__module__}.{base.__qualname__}.{name}", bool(same)]
                break
    return out


def read_at_import(names):
    """True when a tests/ .py file names one of `names` outside every def (module level, class bodies, decorators,
    default values): its import, which the runner parent did with the old values, may have read it."""
    if not _TEST_SOURCES:
        for root, _, files in os.walk(os.path.join(ST, "tests")):
            for nm in files:
                if nm.endswith(".py"):
                    try:
                        with open(os.path.join(root, nm), encoding="utf-8", errors="replace") as fh:
                            _TEST_SOURCES.append(fh.read())
                    except OSError:
                        return True
        _TEST_SOURCES.append("")
    pat = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in sorted(names)) + r")\b")
    for text in _TEST_SOURCES:
        if not pat.search(text):
            continue
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if not pat.search(line):
                continue
            ind, in_def = len(line) - len(line.lstrip()), False
            for j in range(i - 1, -1, -1):
                st = lines[j].lstrip()
                if not st or st.startswith("#"):
                    continue
                k = len(lines[j]) - len(st)
                if k < ind:
                    if st.startswith(("def ", "async def ")):
                        in_def = True
                        break
                    if k == 0:
                        break
                    ind = k
            if not in_def:
                return True
    return False


def set_plan(r):
    """[(module, class path, name, value)] for a key check whose every difference was handed back, or
    None."""
    vals = r.get("values") or {}
    out = []
    for k, mod, gone, _, b in r["diffs"]:
        if gone or k not in vals:
            return None
        v = ast.literal_eval(vals[k])
        if b[1] == "G":
            out.append((b[2], (), k[len(b[2]) + 1:], v))
        else:
            out.append((b[2], tuple(b[3].split(".")), k[len(b[2]) + len(b[3]) + 2:], v))
    if not out or read_at_import({x[2] for x in out}):
        return None
    return out


def key_check():
    """Child process: stdin is {path: patched text}. Prints the changed holders and, per dict / set holder, the
    changed keys, as JSON."""
    files = json.loads(sys.stdin.read())["files"]
    sys.dont_write_bytecode = True
    print(json.dumps(key_snap(files)))
    return 0


def def_readers(m, mod, qual, name, depth=0):
    """The tests that ran the body of `def name` written in module `mod` (in class path `qual`, or at module level
    when qual is "", also through one `from X import name`), from the map's run spans, as a set; None when no such
    def is written there."""
    base = mod.replace(".", "/")
    path = next((p for p in (base + ".py", base + "/__init__.py") if p in m.d["files"]), None)
    src = m.source(path) if path else None
    if src is None:
        return None
    try:
        body = ast.parse(src).body
    except (SyntaxError, ValueError):
        return None
    for q in qual.split(".") if qual else ():
        c = next((s for s in body if isinstance(s, ast.ClassDef) and s.name == q), None)
        if c is None:
            return None
        body = c.body
    defs = [s for s in body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == name]
    if not defs:
        if qual or depth:
            return None
        for s in body:
            if not isinstance(s, ast.ImportFrom):
                continue
            orig = next((x.name for x in s.names if (x.asname or x.name) == name), None)
            if orig is None:
                continue
            pkg = mod if path.endswith("/__init__.py") else mod.rpartition(".")[0]
            for _ in range(max(s.level - 1, 0)):
                pkg = pkg.rpartition(".")[0]
            src_mod = (pkg + "." + s.module if s.module else pkg) if s.level else s.module
            return def_readers(m, src_mod, "", orig, 1) if src_mod else None
        return None
    rt, sel = m.spans(path, "rt"), 0
    for d in defs:
        for x in range(d.body[0].lineno, d.end_lineno + 1):
            r = rt.find(x)
            if r is not None:
                sel |= m.set(r[2])
    return {n for n in range(m.n) if sel >> n & 1}


def key_readers(m, diffs, by_key, shadow=None, renamed=None):
    """The tests that read an entry the change touched (also its size or key order where those changed), the tests
    the map could not follow, or a string saying why the change cannot be narrowed.
    A class constant a class gained (`shadow`: what it hides) is read wherever the hidden one was read through the
    class (the map's reads count every lookup that reached the hidden one); one with the same value changes no read.
    A class constant a class lost was read only where its own reads were."""
    km = m.d["keys"]
    shadow = shadow or {}
    renamed_tests = set()
    if renamed:
        # a test file that names an old name may import it with the module (then all its tests fail) or use it
        # without a call: all its tests; a helper under tests/ that names one: no narrowing
        pat = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in sorted(set(renamed.values()))) + r")\b")
        for p in m.d.get("src", {}):
            if p.startswith("tests/") and p.endswith(".py") and pat.search(m.source(p) or ""):
                sid = m.d["file_tests"].get(p)
                if sid is None:
                    return f"a test helper names a renamed name: {p}"
                mask = m.set(sid)
                renamed_tests.update(n for n in range(m.n) if mask >> n & 1)
    cache = km.setdefault("_r", {})
    keep = set(km["u"])

    def table(tid):
        r = cache.get(tid)
        if r is None:
            r = cache[tid] = json.loads(km["t"][str(tid)])
        return r
    if any(k.startswith("IMPORTERROR") and gone for k, _, gone, _, _ in diffs):
        return "a module that failed to import now imports"
    # Modules that no longer import: every test that imports one (an import statement or importlib, at the test
    # module's import or while it runs) or looks its dialect up by name, which imports the module.
    failed = {mod for k, mod, gone, _, _ in diffs if k.startswith("IMPORTERROR")}
    # A class defined elsewhere that the walk reached through a failed module is now named by another path: the
    # same record under a new key is the same, unchanged value.
    def tail(k):  # the attribute part of a key (records name the path's module and class, not the owner's)
        return "::" + k.rsplit("::", 1)[1] if "::" in k else k.rsplit(".", 1)[-1].rsplit(":", 1)[-1]
    gone_recs = {(tail(k), a[0], a[1]) for k, _, gone, a, _ in diffs if gone and a is not None}
    dialects = set()
    if failed:
        imp = km.get("_imp")
        if imp is None:
            imp = km["_imp"] = json.loads(km["imp"])
        for name, ns in imp.items():
            if any(name == f or name.startswith(f + ".") for f in failed):
                keep.update(ns)
        reg = km["h"].get(DIALECT_TABLE)
        for f in failed:
            parts = f.split(".")
            if len(parts) == 3 and parts[:2] == ["sqlglot", "dialects"]:
                if reg is None:
                    return "no dialect registry"
                dialects.add("s:" + parts[2])
                r = table(reg)
                for sk in ("s:" + parts[2], "#w", "#k", "#l", "#m"):
                    keep.update(r.get(sk, ()))
    for k, mod, gone, a, b in diffs:
        if k.startswith("IMPORTERROR") or mod in failed:
            continue  # the module's own state went with it
        if failed and gone and any(k.startswith(f + ".") or k.startswith(f + ":") or k.replace(":", ".", 1) == f
                                   for f in failed):
            continue  # reached only through a failed module, or the package's name for it: its users are in
        if k.endswith("::order") and k[: -len("::order")] == mod:
            continue  # the order of a module's globals (names added, moved or gone): nothing reads it at run time
        if k.endswith("::order") and a is not None and b is not None and not failed:
            continue  # the order of a class's own namespace (names added, moved or gone): nothing reads it at run
            # time (sqlglot reads class namespaces only in metaclasses at import, whose results show as other keys)
        if renamed and not failed and (a or b)[1] in ("F", "GF"):
            # a rename to new names: a function that changed or went is read where its pinned body ran (its users
            # call it); one under a new name has no reader but the functions that changed to call it
            name = k.rpartition(":")[2] if (a or b)[1] == "GF" else k.rpartition(".")[2]
            if a is None:
                if name in renamed:
                    continue
                return f"a new function: {k}"
            got = def_readers(m, a[2], a[3], name)
            if got is None:
                return f"a function the map cannot place: {k}"
            keep.update(got)
            continue
        if a is None and b is not None and k in shadow and not failed:
            hid, same = shadow[k]
            if same or (hid is None and renamed and k.rpartition(".")[2] in renamed):
                continue
            st = km["h"].get(hid) if hid else None
            r = table(st) if st is not None and st not in km["no"] else {}
            if "#v" not in r:
                return f"a new class constant hides one the map does not follow: {k}"
            keep.update(r["#v"])
            continue
        if gone and a is not None and a[1] == "V" and not failed:
            st = km["h"].get(k)
            r = table(st) if st is not None and st not in km["no"] else {}
            if "#v" in r:
                keep.update(r["#v"])
                continue
        if failed and a is None and (tail(k), b[0], b[1]) in gone_recs:
            continue
        pkg, _, name = k.partition(":")
        if failed and pkg in LAZY_PKGS and (a is None or b is None) and (a or b)[1].startswith("G") and                 "." not in name and ":" not in name:
            continue  # a lazily cached package attribute (the package's __getattr__ gives it either way)
        ch, tid = by_key.get(k), km["h"].get(k)
        ch = None if ch is None else set(ch)
        if k == DIALECT_TABLE and ch is not None and not ch - dialects - {"#n", "#order"}:
            continue  # only the failed dialects are missing: their users are in
        if ch is None and tid is not None and not gone:
            r = table(tid)
            if "#v" in r:  # a class constant: the tests that read it
                keep.update(r["#v"])
                continue
        if ch is None or tid is None:
            return f"not a followed table: {k}"
        if tid in km["no"]:
            return f"table read where its uses cannot be seen: {k}"
        r = table(tid)
        if "#m" in r or "?" in r or "?" in ch:
            return f"table written at run time or with unnamed keys: {k}"
        keep.update(r.get("#w", ()))
        for sk in ch:
            keep.update(r.get({"#n": "#l", "#order": "#k"}.get(sk, sk), ()))
    keep.update(renamed_tests)
    return keep


def run_warm_checks(jobs, t0):
    """{commit: answer} for jobs that carry "mods", through one key server; none starts after the budget."""
    res = {}
    if not jobs:
        return res
    tmp = tempfile.mkdtemp(prefix="kw")
    try:
        p = subprocess.Popen([sys.executable, "-s", "-P", os.path.abspath(__file__), "--key-server", SQ],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=snap_env(), cwd="/tmp")
        try:
            for i, k in enumerate(sorted(jobs)):
                if time.time() - t0 > KEY_BUDGET_S:
                    break
                out = os.path.join(tmp, f"{i}.json")
                p.stdin.write(json.dumps({"files": jobs[k]["files"], "mods": sorted(jobs[k]["mods"]), "out": out}) + "\n")
                p.stdin.flush()
                if p.stdout.readline().strip() != "done":
                    break
                with open(out) as fh:
                    res[k] = json.load(fh)
        finally:
            try:
                p.stdin.close()
            except OSError:
                pass
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
    except Exception as exc:  # noqa: BLE001
        print(f"warm key checks: {exc!r}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def run_key_checks(jobs, t0):
    """{commit: the key check's answer (dict), or None}. Two at a time (each holds a tree copy and a snapshot, ~16 MB
    of the 64 MB /tmp); none starts after the budget."""
    def one(k):
        if time.time() - t0 > KEY_BUDGET_S:
            return k, None
        try:
            return k, key_snap(jobs[k]["files"])
        except Exception as exc:  # noqa: BLE001
            print(f"{k}: key check: {exc!r}", file=sys.stderr)
            return k, None
    with ThreadPoolExecutor(2) as ex:
        return dict(ex.map(one, sorted(jobs)))


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


# A TPC-DS check whose commit only swaps function bodies runs in a fork of the runner parent, which has the pinned
# tree imported already: the child puts the new code into the live functions as a hot test run does (modules not
# loaded yet come from the patched text), then runs the same stages. Saves the new interpreter, the tree copy and
# the import (~0.5 s CPU a check). Any doubt there: the check runs in its own process as before.
_LIVE = {}  # set by runner_phase once its parent imported the tree: {"plans": {commit: hot plan}, "idx": functions}


def live_check(job, plan, idx, timeout):
    """(True / False, why) from a forked check on the live tree, or (None, why) when it cannot tell."""
    import select
    import signal
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 3
        try:
            os.close(r)
            try:
                os.setpgid(0, 0)
            except OSError:
                pass
            dn = os.open(os.devnull, os.O_WRONLY)
            os.dup2(w, 1)
            if not RUN_DEBUG:
                os.dup2(dn, 2)
            sys.stdout = os.fdopen(1, "w", buffering=1)
            for x, y in plan["pairs"]:
                found = idx.get((x.co_filename, x.co_firstlineno, x.co_name), [])
                live = [f for f in found if f.__code__ == x]
                if len(live) != len(found):
                    print("DOUBT a live function there has other code")
                    os._exit(3)
                for f in live:
                    f.__code__ = y
            py = {p: t for p, t in plan["files"].items() if p.endswith(".py")}
            if py:
                import importlib
                sys.meta_path.insert(0, _PatchedFinder(py))
                importlib.invalidate_caches()
            if plan["files"]:
                serve_patched_files(plan["files"])
            sys.dont_write_bytecode = True
            code = query_check(job, None)
        except BaseException as exc:  # noqa: BLE001
            try:
                print("error", repr(exc)[:300])
            except BaseException:  # noqa: BLE001
                pass
            code = 1
        finally:
            try:
                sys.stdout.flush()
            except BaseException:  # noqa: BLE001
                pass
            os._exit(code)
    os.close(w)
    buf, t_end, died = b"", time.monotonic() + timeout, False
    try:
        while True:
            left = t_end - time.monotonic()
            if left <= 0:
                died = True
                break
            ready, _, _ = select.select([r], [], [], left)
            if not ready:
                continue
            chunk = os.read(r, 65536)
            if not chunk:
                break
            buf += chunk
    finally:
        os.close(r)
        if died:
            for kill in (lambda: os.killpg(pid, signal.SIGKILL), lambda: os.kill(pid, signal.SIGKILL)):
                try:
                    kill()
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        _, st = os.waitpid(pid, 0)
    lines = buf.decode("utf-8", "replace").strip().splitlines()
    if died:
        return False, lines + ["timed out"]
    rc = os.waitstatus_to_exitcode(st)
    if rc == 3 or any(s.startswith("DOUBT") for s in lines):
        return None, lines + ["doubt"]
    return rc == 0 and lines[-1:] == ["same"], lines


def run_checks(jobs, t0):
    """{commit: True when the check proved the test unaffected}. Two at a time; none starts after the budget."""
    done = {}
    for k in sorted(jobs, key=lambda k: jobs[k]["est"]):
        job, plan = jobs[k], _LIVE.get("plans", {}).get(k)
        if plan is None or job.get("kind") == "cls" or time.time() - t0 > CHECK_BUDGET_S:
            continue
        c0 = sum(os.times()[:4])
        ok, why = live_check(job, plan, _LIVE["idx"], max(20.0, 4 * (job["est"] + job.get("xbudget", 0.0))))
        _log(f"{k}: live check {ok} est {job['est']:.2f} cpu {sum(os.times()[:4]) - c0:.2f} mates_failed {k in _MATE_FAILED} {str(why[-1:])[:120]}")
        _TPC_LIVE[k] = ("live", ok, list(why))
        if os.environ.get("BR_CHECK_AUDIT"):  # dev: the check in its own process agrees
            k2, ok2 = own_checks(jobs, t0, [k])[0]
            print(f"CHECK {'OK' if ok is None or ok == ok2 else 'MISMATCH'} {k} live {ok} own {ok2} {why}",
                  file=sys.stderr, flush=True)
        if ok is None:
            continue
        if not ok:
            _TPC_WHY[k] = str(why[-1:])
            print(f"{k}: tpcds check keeps the test (live): {why}", file=sys.stderr)
        done[k] = ok
    rest = [k for k in jobs if k not in done]
    done.update(dict(own_checks(jobs, t0, rest)))
    return done


def own_checks(jobs, t0, keys):
    """[(commit, proved)] for these checks, each in its own process."""
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
        _TPC_LIVE[k] = ("own", ok, lines)
        if not ok:
            _TPC_WHY[k] = f"rc {p.returncode} {lines[-1:]} {p.stderr[-300:]!r}"
            print(f"{k}: tpcds check keeps the test: rc {p.returncode} {lines[-1:]} {p.stderr[-300:]!r}",
                  file=sys.stderr)
        return k, ok

    if not keys:
        return []
    with ThreadPoolExecutor(CHECK_JOBS) as ex:
        return list(ex.map(one, sorted(keys, key=lambda k: jobs[k]["est"])))


def fetch_tree(m, dest):
    """Build time: the whole pinned tree (GitHub archive of the pinned commit) at dest, for the test runs. Every
    sqlglot/ and tests/ Python file must equal the map's source, and the map must know every one of them."""
    import io
    import tarfile
    import urllib.request
    url = f"https://github.com/tobymao/sqlglot/archive/{PIN}.tar.gz"
    data = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            break
        except Exception as exc:  # noqa: BLE001
            print(f"download attempt {attempt + 1}: {exc!r}")
            time.sleep(3 * (attempt + 1))
    if data is None:
        raise RuntimeError("no archive")
    prefix = f"sqlglot-{PIN}/"
    tmp = dest + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = []
        for mem in tf.getmembers():
            if mem.name.rstrip("/") + "/" == prefix:
                continue
            if not mem.name.startswith(prefix):
                raise RuntimeError(f"unexpected member {mem.name}")
            mem.name = mem.name[len(prefix):]
            members.append(mem)
        tf.extractall(tmp, members=members, filter="data")
    for path in m.d["src"]:
        with open(os.path.join(tmp, path), encoding="utf-8", newline="") as fh:
            if fh.read() != m.source(path):
                raise RuntimeError(f"{path} differs from the map")
    for top in ("sqlglot", "tests"):
        for root, _, names in os.walk(os.path.join(tmp, top)):
            for f in names:
                rel = os.path.relpath(os.path.join(root, f), tmp).replace(os.sep, "/")
                if f.endswith(".py") and rel not in m.d["src"]:
                    raise RuntimeError(f"{rel} is not in the map")
    os.rename(tmp, dest)
    print("pinned tree at", dest)


def prepare(dest):
    """Build time: write the pinned sqlglot tree from the map and compile it, for the checks."""
    import compileall
    m = Map(MAP)
    try:
        fetch_tree(m, ST)
    except Exception as exc:  # noqa: BLE001  (no tree: no test runs, the selection as v11)
        print("pinned tree not fetched:", repr(exc))
        shutil.rmtree(ST, ignore_errors=True)
        shutil.rmtree(ST + ".part", ignore_errors=True)
    p = subprocess.run([sys.executable, "-c", "import duckdb, numpy, pandas, dateutil, pytz"], capture_output=True)
    if p.returncode != 0 and os.path.isdir(ST):  # the test modules would not import: no test runs
        print("test libraries missing:", p.stderr[-300:])
        shutil.rmtree(ST, ignore_errors=True)
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
    try:
        with gzip.open(MAP, "rb") as fh:
            d = json.loads(fh.read())
        try:
            if os.path.isdir(ST) and d.get("steps"):
                knots = {}
                for b in d["steps"]["tests"].values():
                    info = json.loads(zlib.decompress(base64.b64decode(b)))["info"]
                    knots.setdefault(f"{info['file']}:{info['line']}", method_knots(info["file"], info["line"]))
                d["steps"]["knots"] = knots
                print("test statement groups:", len(knots))
        except Exception as exc:  # noqa: BLE001  (the runs then work them out themselves)
            print("test statement groups not written:", repr(exc))
        with open(str(MAP) + ".pkl", "wb") as fh:
            pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # noqa: BLE001  (the map is then read from its JSON)
        print("map pickle not written:", repr(exc))


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


# ---------------------------------------------------------------- test runs
# The v11 selection gives the candidates. The cheap candidates of a commit then run here, on the patched code, in a
# forked child of one warm process; a candidate that passes there is left out, one that fails, is not reached, or
# ran in any doubt stays.
# hot:  only function bodies of sqlglot modules change (module and class level code compiles the same): the child
#        puts the new code objects into the live functions. A guard: a test whose traces run a changed statement but
#        that never entered a changed function here is not trusted.
# fresh: any other change of sqlglot/ or tests/ Python: the child drops the loaded sqlglot and tests modules and
#        imports the patched tree from memory.
# Fixture edits reach the tests through open(). Tests that use other processes run their pool tasks in-process
# (use_local_pool). Long tests, tests that read changed files by other means, and tests that do not pass here on the
# pinned tree (map "run" field) never run here.

ST = os.environ.get("BR_ST", "/app/st")  # the whole pinned tree (sqlglot, tests, fixtures), written by --prepare
PIN = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
RUN_DEADLINE_S = float(os.environ.get("BR_RUN_WALL", 150))  # no child runs past this many seconds of the run
HOLD_FLOOR_S = 1.5  # a test runs when its CPU cost is below max(this, RUN_SHARE * (2 s + its recorded seconds))
RUN_SHARE = 0.7
RUN_LEVELS_S = []  # smaller run sets to weigh against the budget's (v23 tried 0, 0.5, 1, 2, 4 s: board CPU +3, sim −0.4)
HOT_BUDGET_S = float(os.environ.get("BR_HOT_BUDGET", 8.0))  # CPU seconds of test runs per commit
FRESH_BUDGET_S = float(os.environ.get("BR_FRESH_BUDGET", 3.0))
FRESH_IMPORT_S = 1.2  # CPU of a fresh child's import of the patched tree (the map's "run.imp" when measured)
AFTER_FAIL = 0.5  # once a test failed, a test runs only when it costs less than this share of its recorded seconds
# The tests a run left out after its first failure run in a second child when they would leave the answer for enough
# seconds: they pass in about three of four such runs.
RERUN = os.environ.get("BR_RERUN", "1") == "1"
RERUN_PASS = 0.75
RERUN_MIN_S = 2.0  # packed seconds those tests add to the answer, at least
LONG_S = 15.0  # never run here: test_execute_tpcds (it has its own proof) and test_max_nodes
TEST_WALL_S = 4.0  # wall seconds a test may take: max(this, 25 x its cost)
SETUP_WALL_S = 12.0  # wall seconds between tests (imports, class fixtures)
NO_RUN_FILES = ("tests/test_dialect_entry_points.py", "tests/test_sqlglotc.py")
NO_RUN_IDS = {"tests/dialects/test_dialect.py::TestDialect::test_lazy_load"}  # runs python in a subprocess
RUN_DEBUG = os.environ.get("BR_RUN_DEBUG")
RUN_PRICE = float(os.environ.get("BR_RUN_PRICE", 1.0))  # score seconds per local CPU second of a run
RUN_CLEAN_P = float(os.environ.get("BR_RUN_CLEAN_P", 0.5))  # weight of the all-pass gain (runs without a failure: 0.17-0.27 of runs; 0.5 keeps every sample gain)
MATE_P = 0.6  # share of commits with test_execute_tpcds a candidate where no classmate fails (gen2k: ~0.7)
MATE_CHEAP_S = 2.0  # classmates that run first, ahead of the budget's order
BROAD_BREAK = 8  # with this many failing tests, test_execute_tpcds stays whatever its classmates did
EXEC_CLASS = "tests/test_executor.py::TestExecutor"
EXEC_TABLE_USERS = {"test_optimized_tpch", "test_execute_tpch", "test_execute_tpcds"}  # the tests that read the TPC
# tables TestExecutor.setUpClass builds (1.4 s); the class's other tests never touch them
EXEC_SETUP_FILES = ("tests/test_executor.py", "tests/helpers.py", "sqlglot/executor/table.py")  # what that setup runs
EXEC_LAZY_NAMES = ("tpch_conn", "tpcds_conn", "tpch_tables", "tpcds_tables", "tpch_sqls", "tpcds_sqls")


def exec_lazy_ok(files):
    """TestExecutor's TPC tables may be built on first use when the patch changes nothing that building runs or
    reads (the setup's own code, the helpers, the executor's Table, the TPC fixtures)."""
    return not any(p in EXEC_SETUP_FILES or p.startswith(MATE_OFF_DIRS) for p in files)


def install_exec_lazy(cls, put):
    """Replace TestExecutor.setUpClass so that the TPC tables are built when a test first reads them. Should that
    build fail, the eager setup would have failed every test of the class: the whole run is then in doubt."""
    real_setup, real_teardown = cls.__dict__["setUpClass"], cls.__dict__.get("tearDownClass")
    state = {"built": False}

    class Lazy:
        def __init__(self, name):
            self.name = name

        def __get__(self, obj, owner):
            if not state["built"]:
                state["built"] = True
                for n in EXEC_LAZY_NAMES:
                    if isinstance(cls.__dict__.get(n), Lazy):
                        delattr(cls, n)
                try:
                    real_setup.__func__(cls)
                except BaseException:
                    put("X lazy TestExecutor setup failed")
                    raise
            return getattr(cls, self.name)

    def set_up(c):
        for n in EXEC_LAZY_NAMES:
            setattr(cls, n, Lazy(n))
        cls.cache = {}

    def tear_down(c):
        if state["built"] and real_teardown is not None:
            real_teardown.__func__(cls)

    cls.setUpClass = classmethod(set_up)
    cls.tearDownClass = classmethod(tear_down)
MATE_LONG_MS = 15000  # classmates are tests recorded at most this long
MATE_OFF_FILES = ("tests/test_executor.py",)  # the test's own code: its classmates say nothing about it
MATE_OFF_DIRS = ("tests/fixtures/optimizer/tpc-ds/", "tests/fixtures/optimizer/tpc-h/")
CO_OPTIMIZED = 0x1


def _log(*a):
    if RUN_DEBUG:
        if os.environ.get("BR_FAULTS"):
            import resource
            se, ch = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
            a = a + (f"[faults self {se.ru_minflt} children {ch.ru_minflt}]",)
        print(*a, file=sys.stderr, flush=True)


_FN_FIELDS = ("co_code", "co_names", "co_varnames", "co_freevars", "co_cellvars", "co_flags", "co_argcount",
              "co_kwonlyargcount", "co_posonlyargcount", "co_exceptiontable", "co_qualname")
_BODY_FIELDS = ("co_code", "co_names", "co_varnames", "co_exceptiontable", "co_flags")


def _is_code(o):
    return isinstance(o, types.CodeType)


def _ident(c):
    return c.co_qualname, c.co_freevars, c.co_name


def consts_match(a, b):
    return type(a) is type(b) and repr(a) == repr(b)


def same_behaviour(x, y):
    """True when two code objects differ at most in line numbers (nested code objects alike)."""
    if any(getattr(x, f) != getattr(y, f) for f in _FN_FIELDS) or len(x.co_consts) != len(y.co_consts):
        return False
    for a, b in zip(x.co_consts, y.co_consts):
        if _is_code(a) != _is_code(b):
            return False
        if not (same_behaviour(a, b) if _is_code(a) else consts_match(a, b)):
            return False
    return True


def line_consts(c):
    """Indices of the constants a class body stores as __firstlineno__ (the class's line, set since Python 3.13):
    a line added or removed above the class changes only that number."""
    import dis
    out, prev = set(), None
    for ins in dis.get_instructions(c):
        if ins.opname in ("STORE_NAME", "STORE_FAST") and ins.argval == "__firstlineno__" and prev is not None and                 prev.opname == "LOAD_CONST":
            out.add(prev.arg)
        prev = ins
    return out


def body_match(a, b, found):
    """True when a module or class body runs the same statements in both versions. The function code objects the
    two bodies make are collected in `found` as (pinned, patched) pairs and judged afterwards."""
    if any(getattr(a, f) != getattr(b, f) for f in _BODY_FIELDS) or len(a.co_consts) != len(b.co_consts):
        return False
    lines = None
    for i, (x, y) in enumerate(zip(a.co_consts, b.co_consts)):
        if _is_code(x) != _is_code(y):
            return False
        if not _is_code(x):
            if not consts_match(x, y):
                if lines is None:
                    lines = line_consts(a)
                if not (i in lines and type(x) is int and type(y) is int):
                    return False
            continue
        is_fn = x.co_flags & CO_OPTIMIZED
        if is_fn != y.co_flags & CO_OPTIMIZED:
            return False
        if not is_fn:  # a class body inside
            if not body_match(x, y, found):
                return False
        elif _ident(x) != _ident(y):
            return False
        else:
            found.append((x, y))
    return True


def differing_functions(found):
    """The (pinned, patched) pairs whose code differs, nested functions included; None when putting the new code
    into the live functions could not give the patched behaviour (a changed docstring, which is bound when the
    function is made, or nested functions that no longer line up)."""
    diff, todo = [], list(found)
    while todo:
        x, y = todo.pop()
        if same_behaviour(x, y):
            continue
        dx, dy = x.co_consts[:1], y.co_consts[:1]
        if dx != dy and any(isinstance(c, str) for c in dx + dy):
            return None
        diff.append((x, y))
        kx = [c for c in x.co_consts if _is_code(c)]
        ky = [c for c in y.co_consts if _is_code(c)]
        if len(kx) != len(ky) or any(_ident(u) != _ident(v) for u, v in zip(kx, ky)):
            return None
        todo.extend(zip(kx, ky))
    return diff


def run_plan(m, text):
    """How a child can reproduce the patched tree: {"mode": "hot" or "fresh", "files": {path: new text, or None
    when deleted}, "pairs": [(old code, new code)], "removed": {path: [old changed lines]}}, or None."""
    import marshal
    if "\\ No newline" in text:
        return None
    files, removed = {}, {}
    for old, new, blocks in parse_patch(text):
        path = old or new
        if not path or (old and new and old != new) or path in files:
            return None
        if not path.startswith(("sqlglot/", "tests/")) or path.rsplit("/", 1)[-1] == "conftest.py":
            return None
        if path.endswith(".py"):
            if old is None:
                t = "\n".join(x for b in blocks for x in b[1]) + "\n"
            elif new is None:
                t = None
            else:
                src = m.source(path)
                t = patched(src, blocks) if src is not None else None
                if t is None:
                    return None
        else:
            if old is None or new is None or not path.startswith("tests/fixtures/"):
                return None
            try:
                with open(os.path.join(ST, path), encoding="utf-8", newline="") as fh:
                    src = fh.read()
            except (OSError, UnicodeDecodeError):
                return None
            if "\r" in src:
                return None
            t = patched(src, blocks)
            if t is None:
                return None
        files[path] = t
        removed[path] = [ln for b in blocks for ln, _ in b[0]]
    if not files:
        return None
    plan = {"mode": "hot", "files": files, "pairs": [], "removed": removed}
    for path, t in files.items():
        if not path.endswith(".py"):
            continue
        src = m.source(path) if path.startswith("sqlglot/") else None
        if src is None or t is None:
            plan["mode"] = "fresh"
            continue
        fn = os.path.join(ST, path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # both as from a .pyc: a set constant read back from one can iterate in another order (and its
                # repr with it) than the one the compiler built
                a = marshal.loads(marshal.dumps(compiled(src, fn)))
                b = marshal.loads(marshal.dumps(compiled(t, fn)))
            except (SyntaxError, ValueError):
                return None
        got = []
        ch = differing_functions(got) if body_match(a, b, got) else None
        if ch is None:
            plan["mode"] = "fresh"
        else:
            plan["pairs"] += ch
    if plan["mode"] == "fresh":
        plan["pairs"] = []
    return plan


class _SourceLoader:
    """Loads one module of the patched tree from its text in memory."""

    def __init__(self, path, text):
        self.path, self.text = path, text

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        import marshal
        raw = self.text.encode("utf-8", "surrogateescape")
        # through marshal, so that constants come out as they do from a .pyc
        exec(marshal.loads(marshal.dumps(compile(raw, self.path, "exec", dont_inherit=True))), module.__dict__)

    def get_source(self, name):
        return self.text

    def get_filename(self, name):
        return self.path


class _PatchedFinder:
    """First on sys.meta_path in a child: sqlglot and tests modules the patch adds or changes come from memory,
    the ones it deletes do not import, all others load from the pinned tree as usual."""

    def __init__(self, files):
        self.files = files

    def find_spec(self, name, path=None, target=None):
        import importlib.util
        if name.partition(".")[0] not in ("sqlglot", "tests"):
            return None
        base = name.replace(".", "/")
        for rel, is_pkg in ((base + "/__init__.py", True), (base + ".py", False)):
            if rel not in self.files:
                if os.path.exists(os.path.join(ST, rel)):
                    return None
                continue
            text = self.files[rel]
            if text is None:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            where = os.path.join(ST, rel)
            return importlib.util.spec_from_file_location(
                name, where, loader=_SourceLoader(where, text),
                submodule_search_locations=[os.path.dirname(where)] if is_pkg else None)
        return None


_NOT_PATCHED = object()


def serve_patched_files(files):
    """Reading a changed file by name (open, io.open, io.open_code) gives its patched text; a deleted file is
    missing. Writes and every other file go to the real functions."""
    import builtins
    import io
    patched = {os.path.join(ST, rel): text for rel, text in files.items()}
    _open0, _open_code0 = builtins.open, io.open_code

    def lookup(file):
        if not isinstance(file, (str, os.PathLike)):
            return _NOT_PATCHED
        try:
            return patched.get(os.path.abspath(os.fspath(file)), _NOT_PATCHED)
        except (TypeError, ValueError):
            return _NOT_PATCHED

    def as_bytes(file, text):
        if text is None:
            raise FileNotFoundError(2, "No such file or directory", str(file))
        return io.BytesIO(text.encode("utf-8", "surrogateescape"))

    def open_patched(file, mode="r", *args, **kwargs):
        text = _NOT_PATCHED if set(mode) & set("wax+") else lookup(file)
        if text is _NOT_PATCHED:
            return _open0(file, mode, *args, **kwargs)
        raw = as_bytes(file, text)
        if "b" in mode:
            return raw
        opt = dict(zip(("buffering", "encoding", "errors", "newline"), args))
        opt.update(kwargs)
        return io.TextIOWrapper(raw, encoding=opt.get("encoding"), errors=opt.get("errors"), newline=opt.get("newline"))

    def open_code_patched(path):
        text = lookup(path)
        return _open_code0(path) if text is _NOT_PATCHED else as_bytes(path, text)

    builtins.open = open_patched
    io.open = open_patched
    io.open_code = open_code_patched


def use_local_pool():
    """Replace concurrent.futures.ProcessPoolExecutor, before the test modules import it, with a stand-in that does
    each task here, the first time its outcome is wanted. What a test can tell apart from real workers stays the
    same: the call and its value go through pickle, a task's exception comes back from result(), and after an
    initializer that raised or a task that exited, the pool refuses all further tasks."""
    import concurrent.futures as cf
    from concurrent.futures.process import BrokenProcessPool

    def unusable():
        return BrokenProcessPool("A process in the process pool was terminated abruptly while the future was "
                                 "running or pending.")

    class _Deferred(cf.Future):
        def __init__(self, owner, call):
            super().__init__()
            self._owner, self._call = owner, call
            self._state = "FINISHED"

        def _settle(self):
            call, self._call = self._call, None
            if call is None:
                return
            if not self._owner._usable():
                self._exception = unusable()
                return
            try:
                blob = pickle.dumps(call)
            except BaseException as e:  # noqa: BLE001  (a real pool reports it on the future too)
                self._exception = e
                return
            try:
                fn, args, kwargs = pickle.loads(blob)
                value = fn(*args, **kwargs)
            except SystemExit:
                self._owner._dead = True
                self._exception = unusable()
                return
            except KeyboardInterrupt:
                raise
            except BaseException as e:  # noqa: BLE001
                self._exception = e
                return
            try:
                self._result = pickle.loads(pickle.dumps(value))
            except BaseException as e:  # noqa: BLE001
                self._exception = e

        def result(self, timeout=None):
            self._settle()
            if self._exception is not None:
                raise self._exception
            return self._result

        def exception(self, timeout=None):
            self._settle()
            return self._exception

        def cancel(self):
            return False

    class LocalPool:
        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
            self._setup = (initializer, initargs)
            self._began = self._dead = self._closed = False

        def _usable(self):
            if not self._began:
                self._began = True
                fn, args = self._setup
                if fn is not None:
                    try:
                        fn(*args)
                    except KeyboardInterrupt:
                        raise
                    except BaseException:  # noqa: BLE001
                        self._dead = True
            return not self._dead

        def submit(self, fn, /, *args, **kwargs):
            if self._dead:
                raise unusable()
            if self._closed:
                raise RuntimeError("cannot schedule new futures after shutdown")
            return _Deferred(self, (fn, args, kwargs))

        def map(self, fn, *iterables, timeout=None, chunksize=1):
            futures = [self.submit(fn, *a) for a in zip(*iterables)]
            return (f.result() for f in futures)

        def shutdown(self, wait=True, *, cancel_futures=False):
            self._closed = True

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.shutdown()
            return False

    cf.ProcessPoolExecutor = LocalPool


def function_index():
    """Live functions of the sqlglot tree by (file, first line, name)."""
    import gc
    import types
    root = os.path.join(ST, "sqlglot", "")
    idx = {}
    for o in gc.get_objects():
        if type(o) is types.FunctionType:
            c = o.__code__
            if c.co_filename.startswith(root):
                idx.setdefault((c.co_filename, c.co_firstlineno, c.co_name), []).append(o)
    return idx


def make_immortal():
    """Give the objects this process holds CPython's never-freed reference count (3.12 and 3.13 on 64-bit skip count
    updates on such objects), so a forked child stops writing to, and so copying, the parent's memory pages only to
    count references. Nothing so marked is freed again; the processes that mark end with os._exit."""
    if (os.environ.get("BR_IMM") == "0" or sys.implementation.name != "cpython"
            or sys.version_info[:2] not in ((3, 12), (3, 13)) or sys.maxsize < 2 ** 63 - 1):
        return 0
    import ctypes
    import gc
    put = ctypes.c_ssize_t.from_address
    objs = gc.get_objects()
    for o in objs:
        put(id(o)).value = 0xFFFFFFFF
    return len(objs)


def all_codes(c):
    out = [c]
    for k in c.co_consts:
        if hasattr(k, "co_code"):
            out += all_codes(k)
    return out


# warm (a fresh plan whose Python edits are in sqlglot modules with a small importer group, see warm_group): the
# child drops only that group (the changed modules and every module that imports one while loading) and all tests
# modules, then imports the group again from the patched tree, as the warm key check does; the rest of sqlglot stays
# as the parent loaded it. Same tree as a fresh child's at ~1/10 of the import CPU.
def fresh_group(plan, graph):
    """The sorted module group a warm child re-imports for this fresh plan, or None (the whole tree then)."""
    if not graph:
        return None
    py = {p: t for p, t in plan["files"].items() if p.endswith(".py")}
    if not py or any(not p.startswith("sqlglot/") or t is None for p, t in py.items()):
        return None
    g = warm_group(graph, py)
    return sorted(g) if g else None


def drop_group(group):
    """In a warm child: forget the group's modules (and what a lazy package handed out of them) and every tests
    module. Returns the group modules that were loaded, to import again once the patched finder is in place."""
    import types
    group = set(group)
    loaded = sorted(m for m in group if m in sys.modules)
    for m in sorted(group, key=len, reverse=True):
        old = sys.modules.pop(m, None)
        parent, _, leaf = m.rpartition(".")
        if old is not None and parent in sys.modules and getattr(sys.modules[parent], leaf, None) is old:
            delattr(sys.modules[parent], leaf)
    for n, mo in list(sys.modules.items()):
        if (n == "sqlglot" or n.startswith("sqlglot.")) and n not in group and "__getattr__" in vars(mo):
            for a, v in list(vars(mo).items()):
                if not isinstance(v, types.ModuleType) and getattr(v, "__module__", None) in group:
                    delattr(mo, a)
    for n in [k for k in sys.modules if k.partition(".")[0] == "tests"]:
        del sys.modules[n]
    return loaded


def stale_dialects(group):
    """True when the dialect registry still holds a class of a group module that is not that module's own now."""
    d = sys.modules.get("sqlglot.dialects.dialect")
    reg = getattr(getattr(d, "_Dialect", None), "_classes", None)
    if not isinstance(reg, dict):
        return True
    group = set(group)
    for v in list(reg.values()):
        mod = getattr(v, "__module__", None)
        if mod in group and getattr(sys.modules.get(mod), getattr(v, "__name__", ""), None) is not v:
            return True
    return False


def child_run(plan, order, tests, wfd, idx, skip_after, red=None):
    """In the forked child: reproduce the patched tree, run `order` (test numbers, in run order), report lines
    "S n" (started), "V n code reached cpu_us" (code 0 pass, 1 fail, 2 not run or skipped), "X why" (doubt about
    the whole run), "END"."""
    import functools
    import importlib
    import unittest

    def put(s):
        os.write(wfd, (s + "\n").encode())

    try:
        dn = os.open(os.devnull, os.O_WRONLY)
        os.dup2(dn, 1)
        if not RUN_DEBUG:
            os.dup2(dn, 2)
        files = plan["files"] if plan else {}
        codes = []
        again = []
        if plan and plan["mode"] == "fresh" and plan.get("group"):
            again = drop_group(plan["group"])
        elif plan and plan["mode"] == "fresh":
            for name in [k for k in sys.modules if k.partition(".")[0] in ("sqlglot", "tests")]:
                del sys.modules[name]
        elif plan:
            for x, y in plan["pairs"]:
                found = idx.get((x.co_filename, x.co_firstlineno, x.co_name), [])
                live = [f for f in found if f.__code__ == x]
                if len(live) != len(found):
                    put("X a live function there has other code")
                    os._exit(0)
                for f in live:
                    f.__code__ = y
            for x, y in plan["pairs"]:
                codes += all_codes(y)
            for mod, path, name, val in plan.get("sets", ()):
                o = sys.modules.get(mod)
                if o is None:
                    continue  # not imported yet: the patched text is served when it is
                for q in path:
                    o = getattr(o, q, None)
                if path and (not isinstance(o, type) or o.__module__ != mod or o.__qualname__ != ".".join(path) or
                             type(o).__setattr__ is not type.__setattr__):
                    put("X an edited constant has no plain class")
                    os._exit(0)
                setattr(o, name, val)
        py = {p: t for p, t in files.items() if p.endswith(".py")}
        if py:
            sys.meta_path.insert(0, _PatchedFinder(py))
            importlib.invalidate_caches()
        if files:
            serve_patched_files(files)
        failed = []
        for name in again:  # what the parent had loaded of the group, now from the patched tree
            try:
                importlib.import_module(name)
            except BaseException:  # noqa: BLE001
                failed.append(name)
        if plan and plan["mode"] == "fresh" and plan.get("group") and (failed or stale_dialects(plan["group"])):
            # a group module that no longer imports, or a dialect registered by the parent's copy that the new
            # import did not replace: the parent's objects would stay reachable. The whole tree then, as fresh.
            for name in [k for k in sys.modules if k.partition(".")[0] in ("sqlglot", "tests")]:
                del sys.modules[name]
        cur = [None, False]
        mon = sys.monitoring
        if codes:
            mon.use_tool_id(mon.PROFILER_ID, "br-guard")

            def on_start(code, offset):
                cur[1] = True
                return mon.DISABLE

            mon.register_callback(mon.PROFILER_ID, mon.events.PY_START, on_start)
            for c in {id(c): c for c in codes}.values():
                mon.set_local_events(mon.PROFILER_ID, c, mon.events.PY_START)
        failed_import = {}
        failed_once = [False]
        suite = unittest.TestSuite()
        for n in order:
            path, cls_name, meth = tests[n]["id"].split("::")
            name = path[:-3].replace("/", ".")
            try:
                if name in failed_import:
                    raise failed_import[name]
                try:
                    mod = importlib.import_module(name)
                except Exception as e:  # noqa: BLE001
                    failed_import[name] = e
                    raise
                case = getattr(mod, cls_name)(meth)
            except Exception:  # noqa: BLE001  (the module does not import, or the test is gone)
                put(f"V {n} 1 0 0")
                failed_once[0] = True
                continue
            case._br_n = n
            if red and n in red:
                try:
                    bm = trimmed_method(case, meth, red[n])
                except Exception:  # noqa: BLE001  (the whole method then runs, with the fixture entries it keeps)
                    bm = None
                if bm is not None:
                    setattr(case, meth, bm)
            if n in skip_after:
                real = getattr(case, meth)

                def method(*a, _real=real, _case=case, **k):
                    if failed_once[0]:
                        _case._br_skip = True
                        raise unittest.SkipTest("after a failure")
                    return _real(*a, **k)

                setattr(case, meth, functools.wraps(real)(method))
            suite.addTest(case)
        if plan and plan["mode"] == "fresh":
            import gc
            gc.freeze()
        te = sys.modules.get("tests.test_executor")
        if (exec_lazy_ok(files) and te is not None and isinstance(getattr(te, "TestExecutor", None), type)
                and "setUpClass" in te.TestExecutor.__dict__
                and not any(tests[n]["id"].rsplit("::", 1)[1] in EXEC_TABLE_USERS for n in order
                            if tests[n]["id"].startswith(EXEC_CLASS + "::"))):
            install_exec_lazy(te.TestExecutor, put)

        class _Tally(unittest.TestResult):
            """Reports each test as it ends: pass, fail, or not run (skipped)."""

            def __init__(self):
                super().__init__()
                # subTest checks this flag: the first failing subtest then ends its test, which is all we need
                self.failfast = True
                self._marks = (0, False, 0.0)  # problems counted before the test, skipped, CPU at start

            def stop(self):
                pass  # with failfast set, unittest would call this and end the suite; every test must still run

            def _exc_info_to_string(self, err, test):
                return ""  # the outcome is all that is reported, so the traceback text is never built

            def _problems(self):
                return sum(map(len, (self.failures, self.errors, self.unexpectedSuccesses)))

            def startTest(self, test):
                super().startTest(test)
                cur[0], cur[1] = test._br_n, False
                spec = red.get(test._br_n) if red else None
                _ENTRY_KEEP[0] = {f: set(v) for f, v in spec["files"].items()} if spec and spec["files"] else None
                if codes:
                    mon.restart_events()
                put(f"S {test._br_n}")
                self._marks = (self._problems(), False, time.process_time())

            def addSkip(self, test, reason):
                super().addSkip(test, reason)
                self._marks = self._marks[:1] + (True,) + self._marks[2:]

            def addError(self, test, err):
                super().addError(test, err)
                if not hasattr(test, "_br_n"):  # setUpClass / setUpModule (or their teardown) raised
                    put("X fixture error")

            def stopTest(self, test):
                super().stopTest(test)
                _ENTRY_KEEP[0] = None
                before, skipped, t0 = self._marks
                code = 1 if self._problems() > before else 2 if skipped else 0
                if code == 1:
                    failed_once[0] = True
                put(f"V {test._br_n} {code} {int(cur[1])} {int((time.process_time() - t0) * 1e6)}")
                cur[0] = None

        suite.run(_Tally())
        put("END")
    except BaseException as e:  # noqa: BLE001
        try:
            put("X " + repr(e)[:200].replace("\n", " "))
        except OSError:
            pass
    finally:
        os._exit(0)


def fork_run(plan, order, tests, idx, skip_after, allowed, cap, wall_left, red=None):
    """Fork, run, collect. ({n: (code, reached, cpu)}, status END / TIMEOUT / DIED, doubt or None)."""
    import select
    import signal
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            os.setpgid(0, 0)
        except OSError:
            pass
        child_run(plan, order, tests, w, idx, skip_after, red)
        os._exit(0)
    os.close(w)
    verdicts, status, doubt = {}, "DIED", None
    now = time.monotonic()
    t_end, t_next, t_cap = now + wall_left, now + SETUP_WALL_S, None
    buf = b""
    try:
        while True:
            now = time.monotonic()
            lim = min(t_end, t_next, t_cap or t_end)
            if now >= lim:
                status = "TIMEOUT"
                break
            ready, _, _ = select.select([r], [], [], lim - now)
            if not ready:
                continue
            chunk = os.read(r, 65536)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                s = line.decode("utf-8", "replace")
                if s == "END":
                    status = "END"
                elif s.startswith("X"):
                    doubt = doubt or s
                elif s.startswith("S "):
                    n = int(s[2:])
                    if t_cap is None and cap:
                        t_cap = time.monotonic() + cap
                    t_next = time.monotonic() + allowed.get(n, TEST_WALL_S)
                elif s.startswith("V "):
                    _, n, code, reached, us = s.split()
                    verdicts[int(n)] = (int(code), reached == "1", int(us) / 1e6)
                    t_next = time.monotonic() + SETUP_WALL_S
    finally:
        os.close(r)
        for kill in (lambda: os.killpg(pid, signal.SIGKILL), lambda: os.kill(pid, signal.SIGKILL)):
            try:
                kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass
        os.waitpid(pid, 0)
        try:
            while os.waitpid(-1, os.WNOHANG)[0] > 0:  # anything a test started (PID 1 in the container)
                pass
        except ChildProcessError:
            pass
    return verdicts, status, doubt


def sim_cost(ts, ms):
    if not ts:
        return 0.0
    w = pack(list(ts), ms)
    return (max(sum(ms[t] for t in x) for x in w) + WORKER_MS * len(w)) / 1000


# Trimmed runs. What a costly test does splits into steps: each top-level statement of its method, and each entry a
# fixture loader of tests.helpers hands it (work a pool task does is counted with the entry that queued it). The map
# lists, per step, the sqlglot code objects the step ran on the pinned tree, and what ran outside every step. When
# none of the functions a hot commit swaps ran outside the steps, a step that never ran one of them does at the
# patched commit what it did at the pinned one, where the test passes; the test then runs with the steps that ran
# one, the statements that share a local name with those (a value one makes and another reads or changes), and the
# statements that may change anything else (attributes, globals, other modules, definitions, returns).
TRIM = not os.environ.get("BR_WHOLE_TESTS")
_ENTRY_KEEP = [None]  # in a run child: {fixture file: entry indices to yield} for the test that runs now
_STMT_HARD_CALLS = {"setattr", "delattr", "exec", "eval", "globals", "locals", "vars", "__import__", "super"}
_STMT_MUTATING = {"pop", "popitem", "update", "add", "clear", "remove", "discard", "setdefault", "extend", "append",
                  "insert", "sort", "reverse", "set", "difference_update", "intersection_update", "setLevel",
                  "addHandler", "removeHandler", "__setitem__", "__delitem__", "__setattr__"}
_AST_CACHE = {}


def step_record(m, tid):
    pm = m.d.get("steps")
    if not pm:
        return None
    dec = pm.setdefault("_dec", {})
    if tid not in dec:
        b = pm["tests"].get(tid)
        r = json.loads(zlib.decompress(base64.b64decode(b))) if b else None
        if r is not None:
            r["_base"] = set(r["base"])
            cb = pm["classes"].get(r["cls"])
            r["_base"] |= set(json.loads(zlib.decompress(base64.b64decode(cb)))) if cb else set()
        dec[tid] = r
    return dec[tid]


def method_node(path, line):
    """The FunctionDef that starts at line of the pinned test file, or None."""
    if path not in _AST_CACHE:
        try:
            with open(os.path.join(ST, path), encoding="utf-8") as fh:
                _AST_CACHE[path] = ast.parse(fh.read())
        except (OSError, SyntaxError, ValueError):
            _AST_CACHE[path] = None
    tree = _AST_CACHE[path]
    if tree is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == line:
            return node
    return None


def stmt_closure(fn, keep):
    """The statements of fn to run so that the statements in keep run as in the whole method, or None when its
    statements cannot be left out."""
    return close_over(stmt_knots(fn), keep)


def close_over(knots, keep):
    """stmt_closure from stmt_knots' answer."""
    if knots is None or knots[1] is None:
        return None
    n, comp, hard = knots
    comps = {comp[i] for i in set(keep) | set(hard)}
    return {i for i in range(n) if comp[i] in comps}


_KNOTS = {}  # "file:line" -> stmt_knots of that test method, written into the map's pickle by --prepare


def method_knots(path, line):
    """(statements, group of each statement, statements that cannot be left out) of the test method at path:line,
    with None groups when its statements cannot be left out; None when there is no such method."""
    k = f"{path}:{line}"
    if k not in _KNOTS:
        node = method_node(path, line)
        _KNOTS[k] = None if node is None else stmt_knots(node)
    return _KNOTS[k]


def stmt_knots(fn):
    """(len(fn.body), [group of each statement], [statements that cannot be left out]), with None groups when the
    statements of fn cannot be left out."""
    n = len(fn.body)
    if isinstance(fn, ast.AsyncFunctionDef) or fn.decorator_list:
        return n, None, []
    nodes = list(ast.walk(fn))
    if any(isinstance(n, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await)) for n in nodes) or any(
            isinstance(n, ast.Attribute) and n.attr.startswith("__") and not n.attr.endswith("__") for n in nodes) or any(
            isinstance(n, ast.Name) and n.id in ("super", "__class__", "locals", "vars", "exec", "eval") for n in nodes):
        return len(fn.body), None, []
    a = fn.args
    local = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    local |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
    for n in nodes:
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            local.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not fn:
            local.add(n.name)
        elif isinstance(n, ast.alias):
            local.add((n.asname or n.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            local.add(n.name)
    if a.args:
        local.discard(a.args[0].arg)  # self: what a statement does to it counts as a change outside (hard)
    droppable = (ast.Expr, ast.With, ast.For, ast.If, ast.Try, ast.Assert, ast.Assign, ast.AnnAssign, ast.AugAssign,
                 ast.Pass, getattr(ast, "TryStar", ast.Try))

    def root(x):
        while isinstance(x, (ast.Attribute, ast.Subscript, ast.Call, ast.Starred)):
            x = x.func if isinstance(x, ast.Call) else x.value
        return x.id if isinstance(x, ast.Name) else None

    parent = list(range(len(fn.body)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner, hard = {}, set()
    for i, st in enumerate(fn.body):
        if not isinstance(st, droppable):
            hard.add(i)
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and n.id in local:
                if n.id in owner:
                    parent[find(i)] = find(owner[n.id])
                else:
                    owner[n.id] = i
            elif isinstance(n, (ast.Return, ast.Delete)):
                hard.add(i)
            elif isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, (ast.Store, ast.Del)):
                if root(n) not in local:
                    hard.add(i)
            elif isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name) and f.id in _STMT_HARD_CALLS:
                    hard.add(i)
                elif isinstance(f, ast.Attribute) and f.attr in _STMT_MUTATING and root(f.value) not in local:
                    hard.add(i)
    return len(fn.body), [find(i) for i in range(len(fn.body))], sorted(hard)


def trims(m, plan, cands, tests):
    """{test: {"frac", "stmts" (statement indices to run, or None for all), "files" ({fixture file: entry indices
    to yield}), "file", "line"}} for the candidates of a hot plan that can run trimmed."""
    if not TRIM or not plan or plan["mode"] != "hot" or not plan["pairs"] or not m.d.get("steps"):
        return {}
    if any(not p.startswith("sqlglot/") or not p.endswith(".py") for p in plan["files"]):
        return {}
    pm = m.d["steps"]
    if not _KNOTS and pm.get("knots"):
        _KNOTS.update(pm["knots"])
    ix = pm.get("_ix")
    if ix is None:
        ix = pm["_ix"] = {nm: i for i, nm in enumerate(pm["names"])}
    changed = set()
    for x, _ in plan["pairs"]:
        rel = os.path.relpath(x.co_filename, ST).replace(os.sep, "/")
        for c in all_codes(x):
            cid = ix.get(f"{rel}:{c.co_firstlineno}:{c.co_qualname}")
            if cid is not None:
                changed.add(cid)
    out = {}
    if not changed:
        return out
    for n in cands:
        tid = tests[n]["id"]
        try:
            r = step_record(m, tid)
            spec = trim_spec(r, changed) if r is not None else None
        except Exception as exc:  # noqa: BLE001
            _log(tid, "trim:", repr(exc))
            spec = None
        if spec is not None:
            out[n] = spec
    return out


def trim_spec(r, changed):
    if changed & r["_base"]:
        return None
    steps = r["steps"]
    keep = 0
    for mask, xs in r["groups"]:
        if not changed.isdisjoint(xs):
            keep |= int(mask, 16)
    if not keep or keep == (1 << len(steps)) - 1:
        return None
    direct, entries = set(), {}
    for i, p in enumerate(steps):
        if keep >> i & 1:
            if p.startswith("s:"):
                direct.add(int(p[2:]))
            else:
                f, j = p[2:].rsplit(":", 1)
                entries.setdefault(f, set()).add(int(j))
    files = r.get("files", {})
    stmts = set(direct)
    filt = {}
    for f, st in files.items():
        if -1 in st:
            return None
        if any(x in direct for x in st):
            continue  # a statement that changed itself loads it: every entry
        filt[f] = entries.get(f, set())
        if filt[f]:
            stmts.update(st)
    if any(f not in files for f in entries):
        return None
    info = r["info"]
    knots = method_knots(info["file"], info["line"])
    if knots is None or knots[0] != info["n"]:
        return None
    run = None if info.get("deco") else close_over(knots, stmts)
    if run is not None and len(run) == info["n"]:
        run = None
    if run is None and not filt:
        return None
    on = set(range(info["n"])) if run is None else run
    t_all = r["base_cost"] + sum(r["cost"])
    t_run = r["base_cost"]
    for i, p in enumerate(steps):
        if p.startswith("s:"):
            t_run += r["cost"][i] if int(p[2:]) in on else 0.0
        else:
            f, j = p[2:].rsplit(":", 1)
            if f in filt:
                t_run += r["cost"][i] if int(j) in filt[f] else 0.0
            elif any(x in on for x in files.get(f, ())):
                t_run += r["cost"][i]
    frac = max(0.02, t_run / t_all) if t_all > 0 else 1.0
    return {"frac": frac, "stmts": None if run is None else sorted(run), "files": {f: sorted(v) for f, v in filt.items()},
            "file": info["file"], "line": info["line"]}


def install_entry_filter():
    """In the parent, before the test modules import: the fixture loaders of tests.helpers yield only the entries a
    trimmed test keeps (all of them outside such a test)."""
    import functools
    import inspect
    try:
        import tests.helpers as H
    except Exception:  # noqa: BLE001
        return False
    if getattr(H, "_br_filter", False):
        return True
    origs = {}
    for name in ("load_sql_fixture_pairs", "load_sql_fixtures"):
        f = getattr(H, name, None)
        if not inspect.isgeneratorfunction(f):
            return False
        origs[name] = f

    def make(orig):
        @functools.wraps(orig)
        def gen(filename):
            keep = _ENTRY_KEEP[0]
            want = keep.get(filename) if keep else None
            if want is None:
                yield from orig(filename)
                return
            for j, item in enumerate(orig(filename)):
                if j in want:
                    yield item
        return gen
    for name, f in origs.items():
        setattr(H, name, make(f))
    H._br_filter = True
    return True


def trimmed_method(case, meth, spec):
    """The test method with only the statements spec keeps, bound to case, or None."""
    if spec.get("stmts") is None:
        return None
    fn = getattr(type(case), meth, None)
    if not isinstance(fn, types.FunctionType) or fn.__code__.co_freevars:
        return None
    path = os.path.join(ST, spec["file"])
    if fn.__code__.co_filename != path or fn.__code__.co_firstlineno != spec["line"]:
        return None
    node = method_node(spec["file"], spec["line"])
    if node is None or node.name != meth:
        return None
    new = copy.copy(node)
    new.body = [node.body[i] for i in spec["stmts"]] or [ast.Pass()]
    new.decorator_list = []
    mod = ast.Module(body=[new], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {}
    exec(compile(mod, path, "exec", dont_inherit=True), fn.__globals__, ns)
    f2 = ns.get(meth)
    if not isinstance(f2, types.FunctionType):
        return None
    f2.__qualname__ = fn.__qualname__
    return types.MethodType(f2, case)


def run_choice(m, plan, cands, tests, ms, info, tp=None, mates_all=0, red=None):
    """(tests to run in run order, tests skipped after a failure, wall allowed per test, cap, CPU estimate, classmates
    of test_execute_tpcds run here), or None when the runs cannot save more than they cost."""
    cpu, cls_cpu, bad = info.get("cpu", {}), info.get("cls", {}), set(info.get("bad", ()))
    if not any(p.endswith(".py") for p in plan["files"]):
        # Only test data changed: the candidates are the readers of the changed entries, which then compare against
        # changed expectations or inputs. Such a run nearly always ends in the failure it was sure to find (8 of 8 on
        # the samples): it would change nothing in the answer and only cost CPU.
        return None
    readers = 0
    for p in plan["files"]:
        # .py files are read through open(), io.open (pathlib) or the import system, all served the patched text
        # (linecache keeps the pinned lines, which only traceback text shows); .sql fixtures through open(). Other
        # files may be read where the overlay does not reach (duckdb reads the csv files itself): readers stay.
        if p in m.d["opens"] and not (p.endswith(".py") or (p.startswith("tests/fixtures/") and p.endswith(".sql"))):
            readers |= m.set(m.d["opens"][p])

    def cost(n):
        c = cpu.get(tests[n]["id"])
        c = c if c is not None else 1.5 * ms[n] / 1000 + 0.01
        return c * red[n]["frac"] if red and n in red else c

    def runnable(n):
        tid = tests[n]["id"]
        return not (tid in bad or tid in NO_RUN_IDS or tid.startswith(NO_RUN_FILES) or tid.count("::") != 2
                    or ms[n] / 1000 > LONG_S or readers >> n & 1)

    lazy = exec_lazy_ok(plan["files"])

    def spend_of(ts):
        classes = {tests[n]["id"].rsplit("::", 1)[0] for n in ts}
        if lazy and not any(tests[n]["id"].startswith(EXEC_CLASS + "::")
                            and tests[n]["id"].rsplit("::", 1)[1] in EXEC_TABLE_USERS for n in ts):
            classes.discard(EXEC_CLASS)  # its tables are never built (install_exec_lazy)
        x = sum(cost(n) for n in ts) + sum(cls_cpu.get(c, 0.0) for c in classes)
        return x + (info.get("imp", FRESH_IMPORT_S) if plan["mode"] == "fresh" else 0.02)

    tp_on = (tp in cands and bool(mates_all)
             and not any(p in MATE_OFF_FILES or p.startswith(MATE_OFF_DIRS) for p in plan["files"]))
    # Classmates of test_execute_tpcds (the other TestExecutor tests, the readers of the TPC query files) that are
    # cheap run first: test_execute_tpcds is left out unless one of the classmates that ran fails (see runner_phase).
    protect = {n for n in cands if tp_on and mates_all >> n & 1 and runnable(n) and cost(n) <= MATE_CHEAP_S}
    run = [n for n in cands if runnable(n) and (n in protect or
                                                cost(n) <= max(HOLD_FLOOR_S, RUN_SHARE * (WORKER_MS / 1000 + ms[n] / 1000)))]
    # (a set plan keeps the fresh budget: it runs what a fresh plan would, without the import)
    budget = HOT_BUDGET_S if plan["mode"] == "hot" and not plan.get("sets") else FRESH_BUDGET_S
    # The run sets weighed: the classmates plus the tests with the best recorded-to-CPU ratio up to each CPU level
    # (the last level is the budget, which takes them all when they fit).
    ranked = sorted((n for n in run if n not in protect), key=lambda n: -(ms[n] / 1000 + 0.002) / (cost(n) + 0.002))
    options, seen = [], set()
    for lv in [x for x in RUN_LEVELS_S if x < budget] + [budget]:
        keep, acc = set(), 0.0
        for n in ranked:
            if acc + cost(n) <= lv:
                acc += cost(n)
                keep.add(n)
        r = [n for n in run if n in keep or n in protect]
        if r and frozenset(r) not in seen:
            seen.add(frozenset(r))
            options.append(r)
    if not options:
        return None
    full = sim_cost(cands, ms)
    mates = sorted(protect)
    scored = []
    for r in options:
        rs = set(r)
        rest = [n for n in cands if n not in rs]
        gain = full - sim_cost(rest, ms)
        if mates:
            gain = MATE_P * (full - sim_cost([n for n in rest if n != tp], ms)) + (1 - MATE_P) * gain
        scored.append((gain - RUN_PRICE * spend_of(r), gain, spend_of(r), r, rest))
    scored.sort(key=lambda x: -x[0])
    # Most commits fail some candidate, and a failing test stays with its worker: expect the all-pass gain only with
    # RUN_CLEAN_P, else the gain left when one test that runs fails (averaged over up to 16 of them). The all-pass
    # margin bounds the expected one, so the options go best margin first until none can win.
    best = None
    for margin, gain, spend, r, rest in scored:
        if margin <= 0 or (best is not None and margin <= best[0]):
            break
        fs = sorted(r, key=lambda n: (ms[n], n))
        if len(fs) > 16:
            fs = [fs[i * (len(fs) - 1) // 15] for i in range(16)]
        one = sum(full - sim_cost(rest + [f], ms) for f in fs) / len(fs)
        net = RUN_CLEAN_P * gain + (1 - RUN_CLEAN_P) * max(0.0, one) - RUN_PRICE * spend
        if net > 0 and (best is None or net > best[0]):
            best = (net, r, spend)
    if best is None:
        _log(f"run_choice: no run set expected to save more than it costs (best margin {scored[0][0]:.2f}, "
             f"{len(options)} options, {len(protect)} classmates)")
        return None
    _, run, spend = best
    skip_after = {n for n in run if cost(n) > max(0.03, AFTER_FAIL * ms[n] / 1000)}
    first = {}
    for n in sorted(run):
        first.setdefault(tests[n]["id"].rsplit("::", 1)[0], n)
    # One class after another (class fixtures run once); in a class, the tests a failure would skip come last.
    run.sort(key=lambda n: (first[tests[n]["id"].rsplit("::", 1)[0]], n in skip_after, n))
    allowed = {n: max(TEST_WALL_S, 25 * cost(n)) for n in run}
    cap = 2.0 + 3.0 * spend + max(allowed.values())
    return run, skip_after, allowed, cap, spend, mates


def expected_callers(m, plan):
    """Tests whose traces ran a changed statement of a hot commit's sqlglot modules that the parent has loaded."""
    exp_mask = 0
    for x, _ in plan["pairs"]:
        rel = os.path.relpath(x.co_filename, ST).replace(os.sep, "/")
        mod = rel[:-3].replace("/", ".").removesuffix(".__init__")
        if mod not in sys.modules or rel not in m.d["files"]:
            continue
        lines = {ln for c in all_codes(x) for _, _, ln in c.co_lines() if ln is not None}
        last = max(lines)
        for ln in plan["removed"].get(rel, ()):
            if x.co_firstlineno < ln <= last:
                sp = line_span(m, rel, ln)
                if sp and x.co_firstlineno < sp[0] and sp[1] <= last:
                    exp_mask |= sp[2]
    return exp_mask


def tpcds_mates(m, tests, ms, tp):
    """Bit mask of test_execute_tpcds's classmates: the other TestExecutor tests and the tests that read a TPC query
    file, recorded at most MATE_LONG_MS."""
    if tp is None:
        return 0
    readers = 0
    for p, sid in m.d["opens"].items():
        if p.startswith(MATE_OFF_DIRS):
            readers |= m.set(sid)
    mask = 0
    for n, t in tests.items():
        if n != tp and ms[n] <= MATE_LONG_MS and (t["id"].startswith("tests/test_executor.py::TestExecutor::")
                                                  or readers >> n & 1):
            mask |= 1 << n
    return mask


# Classmate run for test_execute_tpcds. On every labelled commit where that test fails (20 of 20: samples, lab190,
# gate2, gen2k), another TestExecutor test fails too when run on the patched tree. So when a commit keeps it with no
# failure seen, the other TestExecutor tests run in a child that imports the patched tree afresh; if every one ends
# and passes, test_execute_tpcds leaves. (The optimizer's TPC tests compare SQL text: they fail on edits whose
# query results stay the same, so they are no evidence; test_optimized_tpch is costly and never needed.)
MATE_CLASS = "tests/test_executor.py::TestExecutor::"
MATE_SKIP_IDS = {"tests/test_executor.py::TestExecutor::test_optimized_tpch"}
# these run first (on the labelled commits, one of them fails whenever test_execute_tpcds fails); the run stops at the
# first failure
MATE_FIRST = ("test_integer_division_truncates_towards_zero", "test_execute_tpch", "test_execute_subqueries",
              "test_static_queries")
MATE_MIN_GAIN_S = 4.0  # seconds that leaving test_execute_tpcds out must save at least
MATE_WALL_S = 140.0  # no classmate run starts after this many seconds of the run


def mates_phase(m, tests, ms, tp, out, no_run, tp_ok, texts, t_start):
    """{commit: True when test_execute_tpcds leaves, False when a classmate failed} for the commits that keep it
    without a failure seen."""
    if tp is None or not os.path.isdir(os.path.join(ST, "sqlglot")) or not hasattr(os, "fork"):
        return {}
    mask = tpcds_mates(m, tests, ms, tp)
    pool = [n for n in sorted(tests) if mask >> n & 1 and tests[n]["id"].count("::") == 2
            and tests[n]["id"] not in NO_RUN_IDS and not tests[n]["id"].startswith(NO_RUN_FILES)]
    pool = [n for n in pool if tests[n]["id"].startswith(MATE_CLASS) and tests[n]["id"] not in MATE_SKIP_IDS]
    rank = {name: i for i, name in enumerate(MATE_FIRST)}
    pool.sort(key=lambda n: (rank.get(tests[n]["id"].rsplit("::", 1)[1], len(rank)), n))
    if not pool:
        return {}
    todo = []
    for c, chosen in out.items():
        seen = _SEEN_FAIL.get(c, set())
        if (tp not in chosen or c in no_run or c in tp_ok or c in _MATE_FAILED or tp in seen
                or len(seen) >= BROAD_BREAK):
            continue
        text = texts.get(c)
        files = set()
        for old, new, _ in parse_patch(text):
            files.update(x for x in (old, new) if x)
        if any(f in MATE_OFF_FILES or f.startswith(MATE_OFF_DIRS) for f in files):
            continue
        if sim_cost(chosen, ms) - sim_cost([n for n in chosen if n != tp], ms) < MATE_MIN_GAIN_S:
            continue
        plan = run_plan(m, text)
        if plan:
            todo.append((c, plan))
    if not todo:
        return {}
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    if os.getcwd() != ST:
        os.chdir(ST)
    if ST not in sys.path:
        sys.path.insert(0, ST)
    import concurrent.futures as cf
    if cf.ProcessPoolExecutor.__name__ != "LocalPool":
        use_local_pool()
    cpu = (m.d.get("run") or {}).get("cpu", {})
    allowed = {n: max(TEST_WALL_S, 25 * cpu.get(tests[n]["id"], 1.5 * ms[n] / 1000)) for n in pool}
    graph = None
    if not os.environ.get("BR_NO_WARM"):
        try:
            with open(SD + ".graph.json") as fh:
                graph = json.load(fh)
        except (OSError, ValueError):
            graph = None
    res = {}
    for c, plan in todo:
        left = t_start + MATE_WALL_S - time.monotonic()
        if left < 10.0:
            break
        # (a warm child: only the changed modules' importer group comes again from the patched tree)
        fp = dict(plan, mode="fresh", pairs=[], group=None, sets=None)
        fp["group"] = fresh_group(fp, graph)
        v, status, doubt = fork_run(fp, pool, tests, _LIVE.get("idx", {}), set(pool), allowed, None, left)
        bad = [n for n, x in v.items() if x[0] == 1]
        if bad:
            res[c] = False
            _MATE_FAILED.add(c)
        elif status == "END" and not doubt and all(n in v and v[n][0] == 0 for n in pool):
            res[c] = True
        _log(f"{c} classmates: {len(v)}/{len(pool)} ran, {len(bad)} fail, status {status}"
             f"{' doubt ' + doubt if doubt else ''} -> {'leaves' if res.get(c) else 'stays'}")
    return res


def runner_phase(m, tests, ms, work, t_start, tp=None, checks=()):
    """work: [(commit, candidates, patch text)]. Returns ({commit: tests that passed here}, {commits whose
    test_execute_tpcds classmates all passed here}). checks: commits with a TPC-DS check, which can then run on
    this process's tree (_LIVE)."""
    import gc
    info = m.d.get("run") or {}
    if not os.path.isdir(os.path.join(ST, "sqlglot")) or not hasattr(os, "fork") or not info.get("cpu"):
        return {}, set()
    mates_all = tpcds_mates(m, tests, ms, tp)
    # a failing TestExecutor test is evidence that test_execute_tpcds fails; a failing TPC test of the optimizer is
    # not (it compares SQL text): with only those failing, the commit goes to the classmate run (mates_phase)
    exec_mates = 0
    for n, t in tests.items():
        if mates_all >> n & 1 and t["id"].startswith(MATE_CLASS) and t["id"] not in MATE_SKIP_IDS:
            exec_mates |= 1 << n
    graph = None
    if not os.environ.get("BR_NO_WARM"):
        try:
            with open(SD + ".graph.json") as fh:
                graph = json.load(fh)
        except (OSError, ValueError):
            graph = None
    jobs, plans = [], {}
    for c, cands, text in work:
        try:
            plan = run_plan(m, text)
            if plan and plan["mode"] == "fresh" and c in _SET_PLANS and all(p.endswith(".py") for p in plan["files"]):
                plan = dict(plan, mode="hot", pairs=[], sets=_SET_PLANS[c])
            if plan and plan["mode"] == "fresh":
                plan["group"] = fresh_group(plan, graph)
            if plan and c in checks:
                plans[c] = plan
            red = trims(m, plan, cands, tests) if plan else {}
            ch = run_choice(m, plan, cands, tests, ms, info, tp, mates_all, red) if plan else None
        except Exception as exc:  # noqa: BLE001
            _log(c, "plan:", repr(exc))
            continue
        if ch is not None:
            jobs.append([c, cands, plan, ch, red])
        else:
            _HOW[c] = "noplan" if not plan else "declined"
            _log(c, "no run:", "no plan" if not plan else f"{plan['mode']} {len(cands)} candidates")
    if not jobs:
        return {}, set()
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    os.chdir(ST)
    sys.path.insert(0, ST)
    use_local_pool()
    # The parent imports the test modules of every run once (sqlglot with them), watching whether a changed
    # function starts or a changed file is opened meanwhile: such a hot commit becomes fresh.
    watch = {}
    for c, plan in [(j[0], j[2]) for j in jobs] + list(plans.items()):
        for x, _ in plan["pairs"]:
            watch.setdefault((x.co_filename, x.co_firstlineno, x.co_name), set()).add(c)
    hit, opened = set(), set()
    mon = sys.monitoring
    mon.use_tool_id(mon.COVERAGE_ID, "br-watch")

    def on_start(code, offset):
        k = (code.co_filename, code.co_firstlineno, code.co_name)
        if k in watch:
            hit.update(watch[k])
        return mon.DISABLE

    def audit(event, args):
        if event == "open" and args and isinstance(args[0], str):
            opened.add(os.path.abspath(args[0]))

    mon.register_callback(mon.COVERAGE_ID, mon.events.PY_START, on_start)
    mon.set_events(mon.COVERAGE_ID, mon.events.PY_START)
    auditing = [True]
    sys.addaudithook(lambda e, a: audit(e, a) if auditing[0] else None)
    mods = sorted({tests[n]["id"].split("::")[0][:-3].replace("/", ".") for j in jobs for n in j[3][0]})
    t_imp = time.process_time()
    filtered = any(j[4] for j in jobs) and install_entry_filter()
    for name in mods:
        try:
            __import__(name)
        except BaseException as exc:  # noqa: BLE001  (its tests fail in the child too, and stay)
            _log("import", name, repr(exc)[:120])
    auditing[0] = False
    mon.set_events(mon.COVERAGE_ID, 0)
    mon.register_callback(mon.COVERAGE_ID, mon.events.PY_START, None)
    mon.free_tool_id(mon.COVERAGE_ID)
    mon.restart_events()
    _log(f"parent import {time.process_time() - t_imp:.2f} s CPU, {len(mods)} test modules, hit {sorted(hit)}")
    for j in jobs:
        plan = j[2]
        if plan["mode"] == "hot" and (j[0] in hit or any(os.path.join(ST, p) in opened for p in plan["files"]
                                                         if not p.endswith(".py"))):
            plan["mode"], plan["pairs"] = "fresh", []
            j[4] = {}
            j[3] = run_choice(m, plan, j[1], tests, ms, info, tp, mates_all)
        elif j[4] and not filtered:
            j[4] = {}
            j[3] = run_choice(m, plan, j[1], tests, ms, info, tp, mates_all)
    for j in jobs:
        if j[3] is None:
            _HOW[j[0]] = "declined"
    jobs = [j for j in jobs if j[3] is not None]
    idx = function_index()
    t_imm = time.process_time()
    kept = make_immortal()
    _log(f"immortal {kept} objects, {time.process_time() - t_imm:.3f} s CPU")
    gc.freeze()
    _LIVE.update(idx=idx, plans={c: p for c, p in plans.items() if p["mode"] == "hot" and not p.get("sets")
                                 and c not in hit and not any(
        os.path.join(ST, f) in opened for f in p["files"] if not f.endswith(".py"))})
    out, tp_ok = {}, set()
    jobs.sort(key=lambda j: j[3][4])
    for c, cands, plan, (run, skip_after, allowed, cap, spend, mates), red in jobs:
        left = t_start + RUN_DEADLINE_S - time.monotonic()
        if left < 5.0:
            _HOW.setdefault(c, "late")
            break
        try:
            exp_mask = expected_callers(m, plan) if plan["mode"] == "hot" else 0
            before = os.times()
            verdicts, status, doubt = fork_run(plan, run, tests, idx, skip_after, allowed, cap, left - 2.0, red)
            after = os.times()
            used = after.children_user + after.children_system - before.children_user - before.children_system
            if os.environ.get("BR_HOT_AUDIT") and plan["mode"] == "hot":  # dev: a fresh run agrees
                fplan = dict(plan, mode="fresh", pairs=[])
                v3, st3, d3 = fork_run(fplan, run, tests, idx, set(), {n: 60.0 for n in run}, None, 120.0, None)
                for n in run:
                    if n in verdicts and n in v3 and 2 not in (verdicts[n][0], v3[n][0]):
                        print(f"HOT {'MISMATCH' if verdicts[n][0] != v3[n][0] else 'OK'} {c} {tests[n]['id']} "
                              f"hot {verdicts[n][0]} fresh {v3[n][0]}", file=sys.stderr, flush=True)
            if os.environ.get("BR_WARM_AUDIT") and plan["mode"] == "fresh" and plan.get("group"):  # dev: a fresh run agrees
                fplan = dict(plan, group=None)
                b2 = os.times()
                v4, st4, d4 = fork_run(fplan, run, tests, idx, skip_after, allowed, cap, 120.0, None)
                a2 = os.times()
                print(f"WARM CPU {c} warm {used:.3f} fresh "
                      f"{a2.children_user + a2.children_system - b2.children_user - b2.children_system:.3f}",
                      file=sys.stderr, flush=True)
                for n in run:
                    if n in verdicts and n in v4 and 2 not in (verdicts[n][0], v4[n][0]):
                        print(f"WARM {'MISMATCH' if verdicts[n][0] != v4[n][0] else 'OK'} {c} {tests[n]['id']} "
                              f"warm {verdicts[n][0]} fresh {v4[n][0]}", file=sys.stderr, flush=True)
                    elif (n in verdicts) != (n in v4) or (n in verdicts and (verdicts[n][0] == 2) != (v4[n][0] == 2)):
                        print(f"WARM DIFFRUN {c} {tests[n]['id']} warm {verdicts.get(n)} fresh {v4.get(n)}",
                              file=sys.stderr, flush=True)
            if os.environ.get("BR_TRIM_AUDIT") and any(n in red for n in run):  # dev: the whole tests agree
                v2, st2, d2 = fork_run(plan, run, tests, idx, set(), {n: 60.0 for n in run}, None, 120.0, None)
                for n in run:
                    if n not in red or n not in v2 or n not in verdicts or verdicts[n][0] == 2 or v2[n][0] == 2:
                        continue
                    if verdicts[n][0] != v2[n][0]:
                        print(f"TRIM MISMATCH {c} {tests[n]['id']} trimmed {verdicts[n][0]} whole {v2[n][0]}",
                              file=sys.stderr, flush=True)
                    else:
                        print(f"TRIM OK {c} {n} {verdicts[n][0]} {verdicts[n][2]:.3f} {v2[n][2]:.3f}",
                              file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001
            _log(c, "run:", repr(exc))
            _HOW[c] = "error"
            continue
        if doubt:
            _log(c, plan["mode"], "doubt:", doubt)
            _HOW[c] = "doubt"
            continue
        _HOW[c] = "warm" if plan["mode"] == "fresh" and plan.get("group") else plan["mode"]
        if mates_all and any(mates_all >> n & 1 for n in verdicts):
            _MATES_RAN.add(c)
        passed = {n for n, (code, reached, _) in verdicts.items()
                  if code == 0 and (reached or not exp_mask >> n & 1)}
        out[c] = passed
        _SEEN_FAIL[c] = {n for n, v in verdicts.items() if v[0] == 1}
        failing = [n for n, v in verdicts.items() if v[0] == 1]
        if any(exec_mates >> n & 1 for n in failing):
            _MATE_FAILED.add(c)
        # test_execute_tpcds stays when a classmate that ran fails, when a classmate's pass is not trusted (a hot
        # run that never reached the changed code it was expected to reach), or when many tests fail. On 106
        # labelled commits that select it, every one of the 13 where it fails also fails a cheap classmate.
        if (mates and status == "END" and all(n in verdicts for n in run) and len(failing) < BROAD_BREAK
                and not any(mates_all >> n & 1 for n in failing) and any(n in passed for n in mates)
                and not any(mates_all >> n & 1 and v[0] == 0 and n not in passed for n, v in verdicts.items())):
            tp_ok.add(c)
        _log(f"{c} {'warm' if plan['mode'] == 'fresh' and plan.get('group') else plan['mode']} ran {len(verdicts)}/{len(run)} of {len(cands)} trimmed "
             f"{sum(1 for n in run if n in red)} pass {len(passed)} "
             f"fail {sum(1 for v in verdicts.values() if v[0] == 1)} status {status} est {spend:.2f} cpu {used:.2f}"
             + (f" mates {len(mates)} {'drop tpcds' if c in tp_ok else 'keep tpcds'}" if mates else ""))
        if RERUN and status == "END" and failing:
            try:
                rerun_skipped(c, cands, plan, run, verdicts, passed, red, tests, ms, info, idx, t_start, tp, tp_ok,
                              exp_mask)
            except Exception as exc:  # noqa: BLE001
                _log(c, "rerun:", repr(exc))
    return out, tp_ok


def rerun_skipped(c, cands, plan, run, verdicts, passed, red, tests, ms, info, idx, t_start, tp, tp_ok, exp_mask):
    """A second child for the tests that the run of commit c left out after its first failure, when leaving out the
    ones that pass is expected to save more than the run costs. Adds to passed (and to _SEEN_FAIL) in place."""
    cpu = info.get("cpu", {})

    def cost(n):
        x = cpu.get(tests[n]["id"])
        x = x if x is not None else 1.5 * ms[n] / 1000 + 0.01
        return x * red[n]["frac"] if red and n in red else x
    skipped = [n for n in run if n not in verdicts and n != tp]
    if not skipped:
        return
    answer = [n for n in cands if n not in passed and not (n == tp and c in tp_ok)]
    full = sim_cost(answer, ms)
    # the cheapest seconds first, as long as each adds to the saving
    order = sorted(skipped, key=lambda n: -(ms[n] / 1000 + 0.002) / (cost(n) + 0.002))
    pick, spend, best = [], 0.02 if plan["mode"] == "hot" else info.get("imp", FRESH_IMPORT_S), None
    for n in order:
        pick.append(n)
        spend += cost(n)
        gain = full - sim_cost([x for x in answer if x not in pick], ms)
        net = RERUN_PASS * gain - RUN_PRICE * spend
        if gain >= RERUN_MIN_S and net > 0 and (best is None or net > best[0]):
            best = (net, list(pick), spend, gain)
    if best is None:
        return
    left = t_start + RUN_DEADLINE_S - time.monotonic()
    if left < 5.0:
        return
    _, todo, spend, gain = best
    allowed = {n: max(TEST_WALL_S, 25 * cost(n)) for n in todo}
    v, status, doubt = fork_run(plan, todo, tests, idx, set(), allowed, 2.0 + 3.0 * spend + max(allowed.values()),
                                left - 2.0, red)
    if doubt:
        return
    ok = {n for n, (code, reached, _) in v.items() if code == 0 and (reached or not exp_mask >> n & 1)}
    passed.update(ok)
    _SEEN_FAIL.setdefault(c, set()).update(n for n, x in v.items() if x[0] == 1)
    _log(f"{c} rerun {len(v)}/{len(todo)} of {len(skipped)} skipped: pass {len(ok)}, status {status}, "
         f"est {spend:.2f} s, gain if all pass {gain:.2f} s")


def measure_main(out_path):
    """Dev, in the image: every test class runs on the pinned tree in a child, as the runs do. Writes the map's "run"
    field: per-test CPU, per-class setup CPU, the tests that do not pass here, the CPU of a fresh import."""
    import gc
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    by_n = {t["number"]: t for t in tests}
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    os.chdir(ST)
    sys.path.insert(0, ST)
    use_local_pool()
    by_class = {}
    for t in tests:
        if t["id"].count("::") == 2:
            by_class.setdefault(t["id"].rsplit("::", 1)[0], []).append(t["number"])
    mods = sorted({c.split("::")[0][:-3].replace("/", ".") for c in by_class})
    t0 = time.process_time()
    for name in mods:
        try:
            __import__(name)
        except BaseException as exc:  # noqa: BLE001
            print("import", name, repr(exc)[:200], file=sys.stderr)
    print(f"parent import {time.process_time() - t0:.2f} s", file=sys.stderr)
    gc.freeze()
    cpu, cls, bad = {}, {}, []
    for c, nums in sorted(by_class.items()):
        before = os.times()
        v, status, doubt = fork_run(None, sorted(nums), by_n, {}, set(), {n: 900.0 for n in nums}, None, 3000.0)
        after = os.times()
        total = after.children_user + after.children_system - before.children_user - before.children_system
        tsum = 0.0
        for n in nums:
            tid = by_n[n]["id"]
            if n in v:
                cpu[tid] = round(v[n][2], 4)
                tsum += v[n][2]
            if n not in v or v[n][0] != 0 or doubt:
                bad.append(tid)
        cls[c] = round(max(0.0, total - tsum), 4)
        print(f"{c}: {len(v)}/{len(nums)} {status} {doubt or ''} total {total:.2f} tests {tsum:.2f} "
              f"bad {sum(1 for n in nums if by_n[n]['id'] in bad)}", file=sys.stderr, flush=True)
    imps = []
    for _ in range(3):
        r, w = os.pipe()
        before = os.times()
        pid = os.fork()
        if pid == 0:
            for name in [k for k in sys.modules if k.partition(".")[0] in ("sqlglot", "tests")]:
                del sys.modules[name]
            import importlib
            importlib.import_module("tests.test_transpile")
            os._exit(0)
        os.close(w)
        os.close(r)
        os.waitpid(pid, 0)
        after = os.times()
        imps.append(after.children_user + after.children_system - before.children_user - before.children_system)
    out = {"cpu": cpu, "cls": cls, "bad": sorted(bad), "imp": round(sorted(imps)[1], 3)}
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    print(f"measured {len(cpu)} tests, {len(bad)} bad, fresh import {out['imp']} s", file=sys.stderr)


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
    # Only the answers may reach stdout: they go to a copy of it, and fd 1 becomes stderr for everything else.
    ans_fd = os.dup(1)
    os.dup2(2, 1)
    t_start = time.monotonic()
    out = {}
    order = []
    try:
        order, out = select_all(t_start)
    except BaseException as exc:  # noqa: BLE001  (never crash a run)
        print(f"selector failed: {exc!r}", file=sys.stderr)
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    ms = {t["number"]: round(t["seconds"] * 1000) for t in tests}
    if not order:
        order = sorted(p.stem for p in (DATA / "commits").glob("*.patch"))
    answers = {}
    for stem in order:
        chosen = out.get(stem)
        answers[stem] = pack(list(ms) if chosen is None else chosen, ms)
    if READOUT:
        try:
            readout(answers, ms, tests)
        except Exception as exc:  # noqa: BLE001
            print(f"readout: {exc!r}", file=sys.stderr)
    lines = [json.dumps({"commit": stem, "workers": answers[stem]}) for stem in order]
    data = ("\n".join(lines) + "\n").encode() if lines else b""
    while data:
        data = data[os.write(ans_fd, data):]
    cg = ""
    try:
        with open("/sys/fs/cgroup/cpu.stat") as fh:
            cg = fh.readline().strip()
    except OSError:
        pass
    print(f"done: {time.monotonic() - t_start:.1f} s wall, {sum(os.times()[:4]):.2f} s CPU {cg}", file=sys.stderr)
    sys.stderr.flush()
    os._exit(0)  # no interpreter teardown (it only costs CPU)


# Measurement build only (None in every scored entry): one commit whose answer is empty gets a single worker of tests
# whose recorded milliseconds add up to a number about this batch, so the simulated total tells it.
READOUT = True


_HOW_CODE = {"hot": 0, "warm": 1, "fresh": 2, "declined": 3}
_TPC_LIVE = {}  # probe: commit -> (where, ok, output lines)
_TPC_JOB = {}  # probe: commit -> the check job's estimate
_JOB_ALL = {}  # probe: commit -> its check job before the filter
_FILT = {}  # probe: commit -> why its job was left out (1 test not in, 2 no_run, 3 seen failing, 4 est over allowed)
_GAIN = {}  # probe: commit -> the seconds leaving the test out saves after the runs


def truth_run(c, tp, mates, tests, ms):
    """Readout only: the verdicts of the classmates and of test_execute_tpcds when they really run on the patched
    tree (a fresh import in a forked child): (tpcds code, {classmate: code}, status)."""
    m, text, t0 = _CTX["m"], _CTX["texts"][c], _CTX["t0"]
    if "idx" not in _LIVE or time.monotonic() - t0 > 110:
        return None, {}, "skip"
    plan = run_plan(m, text)
    if not plan:
        return None, {}, "noplan"
    plan = dict(plan, mode="fresh", pairs=[], group=None, sets=None)
    tmap = {t["number"]: t for t in tests}
    order = [n for n in sorted(mates) if tmap[n]["id"].count("::") == 2 and tmap[n]["id"] not in NO_RUN_IDS
             and not tmap[n]["id"].startswith(NO_RUN_FILES)] + [tp]
    allowed = {n: (150.0 if n == tp else 30.0) for n in order}
    v, status, doubt = fork_run(plan, order, tmap, _LIVE["idx"], set(), allowed, None,
                                max(10.0, 170.0 - (time.monotonic() - t0)))
    return (v[tp][0] if tp in v else None), {n: x[0] for n, x in v.items() if n != tp}, (doubt and "doubt") or status


_WHY_CODES = (("optimize differs", 3), ("duckdb sql differs", 4), ("result differs", 5), ("recursionerror", 6),
              ("timed out", 9), ("parsed query deeper", 11), ("no tables", 12), ("doubt", 8), ("error", 7))
_GAIN_BACK = [0]


def readout_value(answers, ms, tests):
    """Probe 35. V (16 bits) about #1, the .py commit that keeps test_execute_tpcds unseen failing: why its check job
    was left out (3 bits: 0 it was not, 1 test not in after the runs, 2 key-check no_run, 3 seen failing, 4 estimate
    over the allowed, 5 no job at all, 6 a check ran, 7 class job), its estimate (4 bits, 3 s units), the seconds
    leaving the test out saves after the runs (3 bits, 4 s units), then the check FORCED here with a 40 s execution
    budget (answers unchanged): how it ended (4 bits: 2 proved, 3 optimize differs over budget, 4 duckdb sql
    differs, 5 result differs, 6 RecursionError, 7 error, 8 doubt, 9 timed out, 10 own-process failure, 11 parsed
    deeper, 12 no tables, 13 other, 14 not run), a query passed only by the test's comparison (1 bit),
    one failed both (1 bit). The readout worker also carries the seconds every drop by the test's comparison saved."""
    def cost(ws):
        ws = [w for w in ws if w]
        return (max(sum(ms[t] for t in w) for w in ws) + WORKER_MS * len(ws)) if ws else 0
    tp = next((t["number"] for t in tests if t["id"] == TPCDS_ID), None)

    def py_only(c):
        files = set()
        for old, new, _ in parse_patch(_CTX["texts"][c]):
            files.update(x for x in (old, new) if x)
        return all(f.startswith("sqlglot/") and f.endswith(".py") for f in files)
    for k, (_, ok, ln) in _TPC_LIVE.items():
        if ok and any(x.startswith("x ") and x.endswith(" f") for x in ln) and answers.get(k) is not None:
            chosen = [t for w in answers[k] for t in w]
            if tp not in chosen:
                _GAIN_BACK[0] += cost(pack(chosen + [tp], ms)) - cost(answers[k])
                print(f"readout: gain back {k}", file=sys.stderr)
    cands = []
    for c in sorted(answers):
        chosen = [t for w in answers[c] for t in w]
        if tp in chosen and tp not in _SEEN_FAIL.get(c, set()) and py_only(c):
            cands.append(c)
    print(f"readout: candidates {[(c, _FILT.get(c), _TPC_JOB.get(c), _GAIN.get(c)) for c in cands]}", file=sys.stderr)
    if not cands:
        return 0xFFFF
    one = cands[0]
    job = _JOB_ALL.get(one)
    if one in _TPC_LIVE:
        filt = 6
    elif job is None:
        filt = 5
    elif job.get("kind") == "cls":
        filt = 7
    else:
        filt = _FILT.get(one, 0)
    est = _TPC_JOB.get(one, job["est"] if job else 0.0)
    gain = _GAIN.get(one, 0.0)
    wc, lines = 14, []
    if job is not None and job.get("kind") != "cls":
        job = dict(job)
        if tables_ready():
            job["xbudget"] = 40.0
        plan = _LIVE.get("plans", {}).get(one)
        if True:
            try:
                if plan is None or "idx" not in _LIVE:
                    p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--tpcds-check"],
                                       input=json.dumps(job), capture_output=True, text=True, timeout=90)
                    lines = p.stdout.strip().splitlines()
                    ok = p.returncode == 0 and lines[-1:] == ["same"]
                    if not ok and not lines[-1:]:
                        lines = lines + ["own " + p.stderr[-200:]]
                else:
                    ok, lines = live_check(job, plan, _LIVE["idx"], 90.0)
                lines = list(lines)
                if ok:
                    wc = 2
                else:
                    w = " ".join(lines[-1:]).lower()
                    wc = next((code for key, code in _WHY_CODES if key in w), 8 if ok is None else 13)
            except subprocess.TimeoutExpired:
                wc = 9
            except Exception as exc:  # noqa: BLE001
                print(f"readout: forced check {exc!r}", file=sys.stderr)
                wc = 7
            if plan is None:
                print("readout: forced check in its own process", file=sys.stderr)
    xs = [ln.split() for ln in lines if ln.startswith("x ")]
    nf = int(any(x[-1] == "f" for x in xs))
    nn = int(any(x[-1] == "n" for x in xs))
    no = sum(1 for ln in lines if ln.startswith("o "))
    print(f"readout: {one} filt {filt} est {est:.1f} gain {gain:.1f} forced {wc} trees differ {no} executed {len(xs)} "
          f"test-comparison {nf} differs {nn} lines {lines[-6:]}", file=sys.stderr)
    return filt << 13 | min(int(est // 3), 15) << 9 | min(int(gain // 4), 7) << 6 | wc << 2 | nf << 1 | nn


def readout(answers, ms, tests):
    value = readout_value(answers, ms, tests)
    target = 50 + value + _GAIN_BACK[0]  # milliseconds on one added worker
    stem = next((c for c in sorted(answers) if not any(answers[c])), None)
    if stem is None:
        print("readout: no empty answer", file=sys.stderr)
        return
    nums = sorted(n for n in ms if ms[n] > 0)
    reach = [1]  # reach[i] has bit s set when some of nums[:i] add up to s ms
    lim = (1 << (target + 1)) - 1
    for n in nums:
        reach.append((reach[-1] | reach[-1] << ms[n]) & lim)
    if not reach[-1] >> target & 1:
        print(f"readout: {target} ms cannot be made", file=sys.stderr)
        return
    picked, left = [], target
    for i in range(len(nums) - 1, -1, -1):
        if left and not reach[i] >> left & 1:
            picked.append(nums[i])
            left -= ms[nums[i]]
    answers[stem] = [sorted(picked)]
    print(f"readout value {value} target {target} ms on {stem}", file=sys.stderr)


def key_phase(m, tests, ms, rows, out):
    """Narrow the commits that only change values of assignments to the tests that read a changed entry. Returns the
    commits where a module no longer imports: every test left uses it, so runs here would only see failures."""
    km = m.d.get("keys")
    no_run = set()
    if not km:
        return no_run
    ids = "\n".join(t["id"] for t in sorted(tests, key=lambda t: t["number"]))
    if hashlib.sha1(ids.encode()).hexdigest() != km.get("ids") or not os.path.isfile(SD):
        return no_run  # the map names tests by number: only when this run numbers them alike
    if not isinstance(km["no"], set):
        km["no"] = set(km["no"])

    def cost(ts):
        if not ts:
            return 0.0
        w = pack(ts, ms)
        return (max(sum(ms[t] for t in x) for x in w) + WORKER_MS * len(w)) / 1000
    graph = None
    try:
        with open(SD + ".graph.json") as fh:
            graph = json.load(fh)
    except (OSError, ValueError):
        pass
    jobs = {}
    for stem, chosen, text in rows:
        c = cost(chosen)
        if c < KEY_MIN_WARM_S:
            continue
        try:
            files = key_job(m, text)
            if files and graph is not None:
                files["mods"] = warm_group(graph, files["files"])
        except Exception as exc:  # noqa: BLE001
            print(f"{stem}: key job: {exc!r}", file=sys.stderr)
            files = None
        if files and (files.get("mods") is not None or c >= KEY_MIN_S):
            jobs[stem] = files
    _log(f"key checks: {len(jobs)}, warm {sum(1 for j in jobs.values() if j.get('mods') is not None)}")
    if not jobs:
        return no_run
    t0 = time.time()
    res = run_warm_checks({k: j for k, j in jobs.items() if j.get("mods") is not None}, t0)
    # a warm answer that could not compare goes the fresh way, where the gate allows it
    fresh = {k: j for k, j in jobs.items() if (k not in res or "why" in res[k]) and cost(
        next(ch for st, ch, _ in rows if st == k)) >= KEY_MIN_S}
    res.update(run_key_checks(fresh, t0))
    for stem, r in res.items():
        if not r or "why" in r:
            _KEYED[stem] = "a"
            print(f"{stem}: key check keeps all: {r and r.get('why')}", file=sys.stderr)
            continue
        _KEYED[stem] = "a"
        keep = key_readers(m, r["diffs"], r["by_key"], r.get("shadow"), jobs[stem].get("renames"))
        if isinstance(keep, str):
            print(f"{stem}: key check keeps all: {keep}", file=sys.stderr)
            continue
        for path in list(jobs[stem]["files"]) + jobs[stem]["data"]:
            sid = m.d["opens"].get(path)
            if sid is not None:
                mask = m.set(sid)
                keep.update(n for n in out[stem] if mask >> n & 1)
        before = len(out[stem])
        out[stem] = [n for n in out[stem] if n in keep]
        _KEYED[stem] = "n"
        if any(d[0].startswith("IMPORTERROR") for d in r["diffs"]):
            _KEYED[stem] = "b"
            no_run.add(stem)
        elif r["diffs"] and SET_PLANS:
            try:
                sp = set_plan(r)
            except Exception as exc:  # noqa: BLE001
                _log(stem, "set plan:", repr(exc))
                sp = None
            if sp:
                _SET_PLANS[stem] = sp
            _log(stem, "set plan:" if sp else "no set plan:", [d[0] for d in r["diffs"]][:8])
        print(f"{stem}: key check keeps {len(out[stem])} of {before} ({len(r['diffs'])} changed)", file=sys.stderr)
    return no_run


def select_all(t_start):
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
        rows.append((patch.stem, chosen, text))
        if tp is not None and mask >> tp & 1:
            try:
                # (up to the cap here; the seconds that leaving the test out saves are known after the runs)
                job = tpcds_job(m, text, tp, CHECK_EST_CAP_S)
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
    _CTX.update(m=m, texts={stem: text for stem, _, text in rows}, t0=t_start)
    _log(f"phase select: {sum(os.times()[:4]):.2f} s CPU, {len(jobs)} check jobs")
    out = {stem: chosen for stem, chosen, _ in rows}
    no_run = set()
    try:
        no_run = key_phase(m, tests, ms, rows, out)
    except Exception as exc:  # noqa: BLE001  (no narrowing)
        print(f"key checks: {exc!r}", file=sys.stderr)
    _log(f"phase keys: {sum(os.times()[:4]):.2f} s CPU")
    try:
        passed, tp_ok = runner_phase(m, {t["number"]: t for t in tests}, ms,
                                     [(stem, out[stem], text) for stem, _, text in rows if out[stem] and stem not in no_run],
                                     t_start, tp, set(jobs))
    except Exception as exc:  # noqa: BLE001  (no runs: the selection as it is)
        print(f"runner: {exc!r}", file=sys.stderr)
        passed, tp_ok = {}, set()
    _log(f"phase runs: {sum(os.times()[:4]):.2f} s CPU, test_execute_tpcds left out by its classmates in {len(tp_ok)}")
    for stem, ok in passed.items():
        out[stem] = [n for n in out[stem] if n not in ok and not (n == tp and stem in tp_ok
                                                                and not os.environ.get("BR_NO_TP_OK"))]
    # The proof checks first, wherever the runs left test_execute_tpcds in without seeing it fail, also where a
    # classmate failed: on the board a commit fails a classmate while test_execute_tpcds passes.
    for k in jobs:
        _TPC[k] = "f"
        _JOB_ALL[k] = jobs[k]
        _FILT[k] = (1 if tp not in out.get(k, ()) else 2 if k in no_run else 3 if tp in _SEEN_FAIL.get(k, ()) else 0)
    jobs = {k: v for k, v in jobs.items() if tp in out.get(k, ()) and k not in no_run
            and tp not in _SEEN_FAIL.get(k, ())}
    for k in list(jobs):
        job = jobs[k]
        if job.get("kind") == "cls":
            continue
        gain = sim_cost(out[k], ms) - sim_cost([n for n in out[k] if n != tp], ms)
        # A check may cost about what leaving the test out saves now (the board charges ~0.6-0.8 of a local CPU
        # second; a check costs ~0.7 of its estimate). Queries whose optimized tree differs may then be executed.
        allowed = max(CHECK_MAX_EST_S, min(CHECK_EST_CAP_S, CHECK_EST_PER_S * gain))
        _TPC_JOB[k] = job["est"]
        _GAIN[k] = gain
        if job["est"] > allowed:
            _FILT[k] = 4
            del jobs[k]
            continue
        if tables_ready():
            job["xbudget"] = round(max(allowed - job["est"], min(CHECK_X_CAP_S, CHECK_X_PER_S * gain) - job["est"] / 2), 3)
    try:
        dropped = run_checks(jobs, time.time())
    except Exception as exc:  # noqa: BLE001
        print(f"tpcds checks: {exc!r}", file=sys.stderr)
        dropped = {}
    _log(f"phase checks: {sum(os.times()[:4]):.2f} s CPU, {len(jobs)} checks")
    for stem, ok in dropped.items():
        _TPC[stem] = "d" if ok else "k"
        if ok:
            out[stem] = [n for n in out[stem] if n != tp]
    # The classmate runs only where no check ran.
    if not os.environ.get("BR_NO_MATES"):
        try:
            mres = mates_phase(m, {t["number"]: t for t in tests}, ms, tp, out, no_run | set(jobs), tp_ok,
                               {stem: text for stem, _, text in rows}, t_start)
        except Exception as exc:  # noqa: BLE001  (test_execute_tpcds stays)
            print(f"classmates: {exc!r}", file=sys.stderr)
            mres = {}
        for stem, ok in mres.items():
            if ok:
                out[stem] = [n for n in out[stem] if n != tp]
        _log(f"phase classmates: {sum(os.times()[:4]):.2f} s CPU, {sum(mres.values())} of {len(mres)} leave")
    return [stem for stem, _, _ in rows], out


def mates_eval_main(args):
    """Dev only: for every labelled commit that selects test_execute_tpcds, run all its runnable classmates on the
    patched tree (fresh import, forked child) and print their verdicts and CPU with the known tpcds outcome.
    args: SET_DIR (commits/, tests.json, answers.json or tpcds.jsonl) [ONLY ids...]"""
    import gc
    set_dir, only = args[0], set(args[1:])
    m = Map(MAP)
    tl = json.load(open(os.path.join(set_dir, "tests.json")))["tests"]
    tests = {t["number"]: t for t in tl}
    ms = {n: round(t["seconds"] * 1000) for n, t in tests.items()}
    tp = next(n for n, t in tests.items() if t["id"] == TPCDS_ID)
    labels = {}
    if os.path.isfile(os.path.join(set_dir, "answers.json")):
        for c, a in json.load(open(os.path.join(set_dir, "answers.json")))["commits"].items():
            labels[c] = tp in a["failing"]
    else:
        for x in open(os.path.join(set_dir, "tpcds.jsonl")):
            r = json.loads(x)
            if r.get("ran"):
                labels[r["id"]] = bool(r["failing"])
    mates_all = tpcds_mates(m, tests, ms, tp)
    mates = [n for n in sorted(tests) if mates_all >> n & 1 and tests[n]["id"].count("::") == 2
             and tests[n]["id"] not in NO_RUN_IDS and not tests[n]["id"].startswith(NO_RUN_FILES)]
    jobs = []
    for c in sorted(labels):
        if only and c not in only:
            continue
        try:
            text = open(os.path.join(set_dir, "commits", c + ".patch"), encoding="utf-8", errors="replace").read()
            mask = select(m, text)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"c": c, "skip": repr(exc)}), flush=True)
            continue
        if not mask >> tp & 1:
            continue
        files = set()
        for old, new, _ in parse_patch(text):
            files.update(x for x in (old, new) if x)
        plan = run_plan(m, text)
        jobs.append((c, plan, sorted(files)))
    print(json.dumps({"jobs": len(jobs), "mates": len(mates)}), flush=True)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    os.chdir(ST)
    sys.path.insert(0, ST)
    use_local_pool()
    for name in sorted({tests[n]["id"].split("::")[0][:-3].replace("/", ".") for n in mates}):
        try:
            __import__(name)
        except BaseException as exc:  # noqa: BLE001
            print(json.dumps({"import": name, "err": repr(exc)[:100]}), flush=True)
    idx = function_index()
    make_immortal()
    gc.freeze()
    for c, plan, files in jobs:
        row = {"c": c, "tp_fail": labels[c], "files": files, "mode": plan and plan["mode"]}
        if not plan:
            print(json.dumps(dict(row, skip="no plan")), flush=True)
            continue
        fp = dict(plan, mode="fresh", pairs=[], group=None, sets=None)
        t0 = time.monotonic()
        v, status, doubt = fork_run(fp, mates, tests, idx, set(), {n: 30.0 for n in mates}, None, 400.0)
        row.update(status=status, doubt=doubt, wall=round(time.monotonic() - t0, 2),
                   v={str(n): [x[0], round(x[2], 4)] for n, x in v.items()})
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--prepare"]:
        prepare(sys.argv[2])
    elif sys.argv[1:2] == ["--tpcds-check"]:
        sys.exit(tpcds_check())
    elif sys.argv[1:2] == ["--cls-check"]:
        sys.exit(cls_check())
    elif sys.argv[1:2] == ["--key-check"]:
        sys.exit(key_check())
    elif sys.argv[1:2] == ["--key-server"]:
        sys.exit(key_server_main(sys.argv[2:3]))
    elif sys.argv[1:2] == ["--graph"]:
        sys.exit(graph_main(sys.argv[2:4]))
    elif sys.argv[1:2] == ["--key-snap"]:
        sys.exit(key_snap_main(sys.argv[2:4]))
    elif sys.argv[1:2] == ["--snap"]:
        sys.exit(snap_main(sys.argv[2:6]))
    elif sys.argv[1:2] == ["--mates-eval"]:
        mates_eval_main(sys.argv[2:])
    elif sys.argv[1:2] == ["--measure"]:
        measure_main(sys.argv[2])
    else:
        main()
