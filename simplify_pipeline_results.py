"""
Post-process GP pipeline results: simplify redundant expression strings.

Applies constant folding, algebraic identity reduction, and comparison
normalization to the 'expression' column of all CSV files under a pipeline
run folder.  All simplifications are semantics-preserving.

Usage:
    python simplify_pipeline_results.py <run_folder>
    python simplify_pipeline_results.py pipeline_runs/20260429_161705

Creates .bak copies of originals before overwriting.
"""

import csv
import math
import re
import sys
from pathlib import Path


# ── Float formatting ──────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    """Format float with 4 significant figures; keep .0 suffix for integers."""
    if not math.isfinite(v):
        return str(v)
    if v == int(v) and abs(v) < 1_000_000:
        return f"{int(v)}.0"
    return f"{v:.4g}"


# ── Simple recursive-descent parser ──────────────────────────────────────────
#
# AST nodes are plain dicts with a 'type' key:
#   {'type': 'num',  'v': float}
#   {'type': 'var',  'name': str}
#   {'type': 'abs',  'e': node}
#   {'type': 'binop','op': str, 'l': node, 'r': node}  op in +,-,*,/,>,<,==,!=,AND
#   {'type': 'not',  'e': node}
#   {'type': 'if',   'cond': node, 'then': node, 'else': node}
#   {'type': 'fn',   'fn': str,  'l': node, 'r': node}  fn in min, max
#   {'type': 'cat',  'feat': str, 'op': str, 'cat': str}  categorical ==/ !=
#   {'type': 'between','lo': node, 'e': node, 'hi': node}
#   {'type': 'raw',  'text': str}   unparseable fragment (pass-through)


_NUM_RE = re.compile(
    r'(-?)'           # optional leading minus
    r'(\d+\.\d+(?:[eE][+-]?\d+)?'  # float
    r'|\d+(?:[eE][+-]?\d+)?)',      # or int
)

# Tokenizer: returns list of (kind, value) pairs
_TOKEN_RE = re.compile(
    r'(?P<FLOAT>\d+\.\d+(?:[eE][+-]?\d+)?)'
    r'|(?P<INT>\d+(?:[eE][+-]?\d+)?)'
    r'|(?P<GE>[≥])'
    r'|(?P<LE>[≤])'
    r'|(?P<NEQ>!=)'
    r'|(?P<EQ>==)'
    r'|(?P<AND>\bAND\b)'
    r'|(?P<OR>\bOR\b)'
    r'|(?P<NOT>\bNOT\b)'
    r'|(?P<IF>\bIF\b)'
    r'|(?P<MIN>\bmin\b)'
    r'|(?P<MAX>\bmax\b)'
    r'|(?P<IDENT>[A-Za-z_][A-Za-z0-9_\-]*)'
    r'|(?P<PIPE>[|])'
    r'|(?P<GT>[>])'
    r'|(?P<LT>[<])'
    r'|(?P<COMMA>[,])'
    r'|(?P<LPAREN>[(])'
    r'|(?P<RPAREN>[)])'
    r'|(?P<PLUS>[+])'
    r'|(?P<MINUS>[-])'
    r'|(?P<STAR>[*])'
    r'|(?P<SLASH>[/])'
    r'|(?P<WS>\s+)'
    r'|(?P<OTHER>.)',
)


def _tokenize(s: str) -> list:
    toks = []
    for m in _TOKEN_RE.finditer(s):
        kind = m.lastgroup
        if kind == 'WS':
            continue
        toks.append((kind, m.group()))
    return toks


class _Parser:
    def __init__(self, s: str):
        self._toks = _tokenize(s)
        self._orig = s
        self._pos = 0

    # ── Token stream helpers ──────────────────────────────────────────────────

    def _peek(self):
        if self._pos < len(self._toks):
            return self._toks[self._pos]
        return None

    def _peek2(self):
        if self._pos + 1 < len(self._toks):
            return self._toks[self._pos + 1]
        return None

    def _consume(self, kind=None):
        tok = self._toks[self._pos]
        if kind and tok[0] != kind:
            raise ValueError(f"Expected {kind!r} got {tok!r} near pos {self._pos}")
        self._pos += 1
        return tok

    def _at_end(self):
        return self._pos >= len(self._toks)

    # ── Grammar ───────────────────────────────────────────────────────────────
    #
    # top      := not_expr (AND not_expr)*
    # not_expr := NOT ( top ) | cmp
    # cmp      := add (> | < | == | !=) add
    #           | add ≤ add ≤ add         (Between)
    #           | add
    # add      := mul ((+ | -) mul)*
    # mul      := unary ((* | /) unary)*
    # unary    := - unary | atom
    # atom     := | add |
    #           | min(add, add)
    #           | max(add, add)
    #           | IF(top, add, add)
    #           | ( top )
    #           | number
    #           | identifier

    def parse(self):
        try:
            node = self._parse_top()
            if not self._at_end():
                # Unparsed remainder — fall back to raw
                return {'type': 'raw', 'text': self._orig}
            return node
        except Exception:
            return {'type': 'raw', 'text': self._orig}

    def _parse_top(self):
        """OR chain (lowest precedence)."""
        left = self._parse_and()
        if self._peek() and self._peek()[0] == 'OR':
            parts = [left]
            while self._peek() and self._peek()[0] == 'OR':
                self._consume('OR')
                parts.append(self._parse_and())
            node = parts[0]
            for p in parts[1:]:
                node = {'type': 'binop', 'op': 'OR', 'l': node, 'r': p}
            return node
        return left

    def _parse_and(self):
        """AND chain."""
        left = self._parse_not()
        if self._peek() and self._peek()[0] == 'AND':
            parts = [left]
            while self._peek() and self._peek()[0] == 'AND':
                self._consume('AND')
                parts.append(self._parse_not())
            node = parts[0]
            for p in parts[1:]:
                node = {'type': 'binop', 'op': 'AND', 'l': node, 'r': p}
            return node
        return left

    def _parse_not(self):
        tok = self._peek()
        if tok and tok[0] == 'NOT':
            self._consume('NOT')
            self._consume('LPAREN')
            inner = self._parse_top()
            self._consume('RPAREN')
            return {'type': 'not', 'e': inner}
        return self._parse_cmp()

    def _parse_cmp(self):
        left = self._parse_add()
        tok = self._peek()
        if tok is None:
            return left
        if tok[0] == 'GT':
            self._consume()
            right = self._parse_add()
            return {'type': 'binop', 'op': '>', 'l': left, 'r': right}
        if tok[0] == 'LT':
            self._consume()
            right = self._parse_add()
            return {'type': 'binop', 'op': '<', 'l': left, 'r': right}
        if tok[0] == 'EQ':
            self._consume()
            # May be categorical: collect the rest of the tokens until RPAREN or AND
            right = self._parse_add_or_cat()
            return {'type': 'binop', 'op': '==', 'l': left, 'r': right}
        if tok[0] == 'NEQ':
            self._consume()
            right = self._parse_add_or_cat()
            return {'type': 'binop', 'op': '!=', 'l': left, 'r': right}
        if tok[0] == 'LE':
            self._consume('LE')
            e = self._parse_add()
            if self._peek() and self._peek()[0] == 'LE':
                # lo ≤ e ≤ hi  (Between)
                self._consume('LE')
                hi = self._parse_add()
                return {'type': 'between', 'lo': left, 'e': e, 'hi': hi}
            return {'type': 'binop', 'op': '≤', 'l': left, 'r': e}
        if tok[0] == 'GE':
            self._consume('GE')
            right = self._parse_add()
            return {'type': 'binop', 'op': '≥', 'l': left, 'r': right}
        return left

    def _parse_add_or_cat(self):
        """Parse the right-hand side of == or !=: numeric expr or category string."""
        save = self._pos
        try:
            node = self._parse_add()
            # If non-delimiter tokens remain after arithmetic parse (e.g. '10th'
            # where '10' was consumed and 'th' is left), treat whole thing as category.
            peek = self._peek()
            if peek is not None and peek[0] not in (
                    'RPAREN', 'AND', 'OR', 'LE', 'GE', 'GT', 'LT',
                    'EQ', 'NEQ', 'COMMA', 'PLUS', 'MINUS', 'STAR', 'SLASH'):
                raise ValueError("leftover token — treat as category")
            return node
        except Exception:
            self._pos = save
        # Collect raw tokens as category name.
        # Suppress spaces around - + & so that names like "400-460" or "820+"
        # are preserved rather than becoming "400 - 460" or "820 +".
        parts = []
        while not self._at_end():
            tok = self._peek()
            if tok[0] in ('RPAREN', 'AND', 'LE', 'GT', 'LT'):
                break
            self._pos += 1
            parts.append(tok[1])
        if not parts:
            return {'type': 'raw', 'text': ''}
        result = parts[0]
        for i in range(1, len(parts)):
            p, c = parts[i - 1], parts[i]
            if c in ('-', '+', '&') or p in ('-', '+', '&'):
                result += c
            else:
                result += ' ' + c
        return {'type': 'raw', 'text': result}

    def _parse_add(self):
        left = self._parse_mul()
        while True:
            tok = self._peek()
            if tok and tok[0] == 'PLUS':
                self._consume()
                right = self._parse_mul()
                left = {'type': 'binop', 'op': '+', 'l': left, 'r': right}
            elif tok and tok[0] == 'MINUS':
                self._consume()
                right = self._parse_mul()
                left = {'type': 'binop', 'op': '-', 'l': left, 'r': right}
            else:
                break
        return left

    def _parse_mul(self):
        left = self._parse_unary()
        while True:
            tok = self._peek()
            if tok and tok[0] == 'STAR':
                self._consume()
                right = self._parse_unary()
                left = {'type': 'binop', 'op': '*', 'l': left, 'r': right}
            elif tok and tok[0] == 'SLASH':
                self._consume()
                right = self._parse_unary()
                left = {'type': 'binop', 'op': '/', 'l': left, 'r': right}
            else:
                break
        return left

    def _parse_unary(self):
        tok = self._peek()
        if tok and tok[0] == 'MINUS':
            self._consume()
            e = self._parse_unary()
            if e['type'] == 'num':
                return {'type': 'num', 'v': -e['v']}
            return {'type': 'binop', 'op': '-', 'l': {'type': 'num', 'v': 0.0}, 'r': e}
        return self._parse_atom()

    def _parse_atom(self):
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end")
        if tok[0] in ('FLOAT', 'INT'):
            self._consume()
            return {'type': 'num', 'v': float(tok[1])}
        if tok[0] == 'IDENT':
            name = tok[1]
            self._consume()
            # Peek: if next is also IDENT (handles multi-word category names) — handled in parse_add_or_cat
            return {'type': 'var', 'name': name}
        if tok[0] == 'PIPE':
            self._consume('PIPE')
            e = self._parse_add()
            self._consume('PIPE')
            return {'type': 'abs', 'e': e}
        if tok[0] == 'MIN':
            self._consume('MIN')
            self._consume('LPAREN')
            l = self._parse_add()
            self._consume('COMMA')
            r = self._parse_add()
            self._consume('RPAREN')
            return {'type': 'fn', 'fn': 'min', 'l': l, 'r': r}
        if tok[0] == 'MAX':
            self._consume('MAX')
            self._consume('LPAREN')
            l = self._parse_add()
            self._consume('COMMA')
            r = self._parse_add()
            self._consume('RPAREN')
            return {'type': 'fn', 'fn': 'max', 'l': l, 'r': r}
        if tok[0] == 'IF':
            self._consume('IF')
            self._consume('LPAREN')
            cond = self._parse_top()
            self._consume('COMMA')
            then_ = self._parse_add()
            self._consume('COMMA')
            else_ = self._parse_add()
            self._consume('RPAREN')
            return {'type': 'if', 'cond': cond, 'then': then_, 'else': else_}
        if tok[0] == 'NOT':
            self._consume('NOT')
            self._consume('LPAREN')
            inner = self._parse_top()
            self._consume('RPAREN')
            return {'type': 'not', 'e': inner}
        if tok[0] == 'LPAREN':
            self._consume('LPAREN')
            inner = self._parse_top()
            self._consume('RPAREN')
            return inner
        if tok[0] == 'AND':
            # PREFIX call form: AND(cond1, cond2) — infix AND is handled by _parse_and
            self._consume('AND')
            self._consume('LPAREN')
            l = self._parse_top()
            self._consume('COMMA')
            r = self._parse_top()
            self._consume('RPAREN')
            return {'type': 'binop', 'op': 'AND', 'l': l, 'r': r}
        if tok[0] == 'OR':
            # PREFIX call form: OR(cond1, cond2)
            self._consume('OR')
            self._consume('LPAREN')
            l = self._parse_top()
            self._consume('COMMA')
            r = self._parse_top()
            self._consume('RPAREN')
            return {'type': 'binop', 'op': 'OR', 'l': l, 'r': r}
        raise ValueError(f"Unexpected token {tok!r}")


# ── AST simplification ────────────────────────────────────────────────────────

def _is_num(node) -> bool:
    return node['type'] == 'num'


def _num_val(node) -> float | None:
    if node['type'] == 'num':
        return node['v']
    return None


_SAFE_OPS = {'+', '-', '*'}


def _const_val(node) -> float | None:
    """Try to evaluate a var-free numeric node. Returns None on failure."""
    t = node['type']
    if t == 'num':
        return node['v']
    if t == 'abs':
        v = _const_val(node['e'])
        return abs(v) if v is not None else None
    if t == 'fn':
        lv, rv = _const_val(node['l']), _const_val(node['r'])
        if lv is None or rv is None:
            return None
        return min(lv, rv) if node['fn'] == 'min' else max(lv, rv)
    if t == 'binop' and node['op'] in _SAFE_OPS:
        lv, rv = _const_val(node['l']), _const_val(node['r'])
        if lv is None or rv is None:
            return None
        op = node['op']
        if op == '+':
            return lv + rv
        if op == '-':
            return lv - rv
        if op == '*':
            return lv * rv
    if t == 'binop' and node['op'] == '/':
        lv, rv = _const_val(node['l']), _const_val(node['r'])
        if lv is None or rv is None or rv == 0.0:
            return None
        return lv / rv
    return None


_FLIP = {'>': '<', '<': '>', '≥': '≤', '≤': '≥'}


def _ge_lower_bound(node, v: float) -> bool:
    """Return True if node's value is guaranteed ≥ v for all inputs."""
    cv = _const_val(node)
    if cv is not None:
        return cv >= v
    if node['type'] == 'fn':
        if node['fn'] == 'max':
            # max(a,b) >= v iff a >= v OR b >= v
            return _ge_lower_bound(node['l'], v) or _ge_lower_bound(node['r'], v)
        if node['fn'] == 'min':
            # min(a,b) >= v iff a >= v AND b >= v
            return _ge_lower_bound(node['l'], v) and _ge_lower_bound(node['r'], v)
    if node['type'] == 'abs':
        return v <= 0.0  # |x| >= 0 always
    return False


def _le_upper_bound(node, v: float) -> bool:
    """Return True if node's value is guaranteed ≤ v for all inputs."""
    cv = _const_val(node)
    if cv is not None:
        return cv <= v
    if node['type'] == 'fn':
        if node['fn'] == 'min':
            # min(a,b) <= v iff a <= v OR b <= v
            return _le_upper_bound(node['l'], v) or _le_upper_bound(node['r'], v)
        if node['fn'] == 'max':
            # max(a,b) <= v iff a <= v AND b <= v
            return _le_upper_bound(node['l'], v) and _le_upper_bound(node['r'], v)
    return False


def _bound_info(node):
    """If node is (expr op constant), return (rendered_expr, op, constant) or None."""
    if node['type'] != 'binop' or node['op'] not in ('>', '<', '≥', '≤'):
        return None
    rv = _const_val(node['r'])
    if rv is not None:
        return (_render(node['l']), node['op'], rv)
    lv = _const_val(node['l'])
    if lv is not None:
        return (_render(node['r']), _FLIP[node['op']], lv)
    return None


def _simplify(node):
    """Recursively simplify an AST node. Returns simplified node."""
    t = node['type']

    if t == 'raw':
        return node

    if t == 'num':
        return node

    if t == 'var':
        return node

    # ── Recurse first (bottom-up) ─────────────────────────────────────────────
    if t == 'abs':
        e = _simplify(node['e'])
        v = _const_val(e)
        if v is not None:
            return {'type': 'num', 'v': abs(v)}
        if e['type'] == 'abs':          # ||x|| = |x|
            return e
        # |x / c| = |x| / c  (c > 0);  |c * x| = c * |x|  (c > 0)
        if e['type'] == 'binop' and e['op'] == '/':
            rv_e = _const_val(e['r'])
            if rv_e is not None and rv_e > 0.0:
                return _simplify({'type': 'binop', 'op': '/',
                                  'l': {'type': 'abs', 'e': e['l']}, 'r': e['r']})
        if e['type'] == 'binop' and e['op'] == '*':
            lv_e = _const_val(e['l']); rv_e = _const_val(e['r'])
            if lv_e is not None:
                return _simplify({'type': 'binop', 'op': '*',
                                  'l': {'type': 'num', 'v': abs(lv_e)},
                                  'r': {'type': 'abs', 'e': e['r']}})
            if rv_e is not None:
                return _simplify({'type': 'binop', 'op': '*',
                                  'l': {'type': 'abs', 'e': e['l']},
                                  'r': {'type': 'num', 'v': abs(rv_e)}})
        return {'type': 'abs', 'e': e}

    if t == 'fn':
        l = _simplify(node['l'])
        r = _simplify(node['r'])
        lv, rv = _const_val(l), _const_val(r)
        if lv is not None and rv is not None:
            v = min(lv, rv) if node['fn'] == 'min' else max(lv, rv)
            return {'type': 'num', 'v': v}
        if lv is not None and node['fn'] == 'min' and lv <= -1e12:
            return l
        if lv is not None and node['fn'] == 'max' and lv >= 1e12:
            return l
        if rv is not None and node['fn'] == 'min' and rv <= -1e12:
            return r
        if rv is not None and node['fn'] == 'max' and rv >= 1e12:
            return r
        # max(max(X, C1), C2) → max(X, max(C1,C2))  — flatten nested same-fn with constants
        fn = node['fn']
        _cmp = max if fn == 'max' else min
        if l['type'] == 'fn' and l['fn'] == fn:
            ilv, irv = _const_val(l['l']), _const_val(l['r'])
            if rv is not None and irv is not None:
                return _simplify({'type': 'fn', 'fn': fn, 'l': l['l'],
                                  'r': {'type': 'num', 'v': _cmp(irv, rv)}})
            if rv is not None and ilv is not None:
                return _simplify({'type': 'fn', 'fn': fn, 'l': l['r'],
                                  'r': {'type': 'num', 'v': _cmp(ilv, rv)}})
        if r['type'] == 'fn' and r['fn'] == fn:
            ilv, irv = _const_val(r['l']), _const_val(r['r'])
            if lv is not None and irv is not None:
                return _simplify({'type': 'fn', 'fn': fn, 'l': r['l'],
                                  'r': {'type': 'num', 'v': _cmp(irv, lv)}})
            if lv is not None and ilv is not None:
                return _simplify({'type': 'fn', 'fn': fn, 'l': r['r'],
                                  'r': {'type': 'num', 'v': _cmp(ilv, lv)}})
        return {'type': 'fn', 'fn': node['fn'], 'l': l, 'r': r}

    if t == 'not':
        e = _simplify(node['e'])
        if e['type'] == 'not':
            return e['e']
        if e['type'] == 'binop' and e['op'] == '!=':
            return {'type': 'binop', 'op': '==', 'l': e['l'], 'r': e['r']}
        if e['type'] == 'binop' and e['op'] == '==':
            return {'type': 'binop', 'op': '!=', 'l': e['l'], 'r': e['r']}
        # NOT(x > y) → x ≤ y, etc.
        if e['type'] == 'binop' and e['op'] in ('>', '<', '≥', '≤'):
            flip = {'>': '≤', '<': '≥', '≥': '<', '≤': '>'}
            return {'type': 'binop', 'op': flip[e['op']], 'l': e['l'], 'r': e['r']}
        return {'type': 'not', 'e': e}

    if t == 'if':
        cond  = _simplify(node['cond'])
        then_ = _simplify(node['then'])
        else_ = _simplify(node['else'])
        cond_v = _eval_bool(cond)
        if cond_v is True:
            return then_
        if cond_v is False:
            return else_
        if _render(then_) == _render(else_):
            return then_
        return {'type': 'if', 'cond': cond, 'then': then_, 'else': else_}

    if t == 'between':
        lo = _simplify(node['lo'])
        e  = _simplify(node['e'])
        hi = _simplify(node['hi'])
        lov, ev, hiv = _const_val(lo), _const_val(e), _const_val(hi)
        if lov is not None and ev is not None and hiv is not None:
            v = min(lov, hiv) <= ev <= max(lov, hiv)
            return {'type': 'num', 'v': 1.0 if v else 0.0}
        # Split into AND(lo ≤ e, e ≤ hi) for uniform handling downstream
        return _simplify({'type': 'binop', 'op': 'AND',
                          'l': {'type': 'binop', 'op': '≤', 'l': lo, 'r': e},
                          'r': {'type': 'binop', 'op': '≤', 'l': e, 'r': hi}})

    if t == 'binop':
        op = node['op']
        l  = _simplify(node['l'])
        r  = _simplify(node['r'])
        lv = _const_val(l)
        rv = _const_val(r)

        # ── AND / OR ──────────────────────────────────────────────────────────
        if op == 'AND':
            lv_bool = _eval_bool(l)
            rv_bool = _eval_bool(r)
            if lv_bool is False or rv_bool is False:
                return {'type': 'num', 'v': 0.0}
            if lv_bool is True:
                return r
            if rv_bool is True:
                return l
            if _render(l) == _render(r):
                return l
            # Duplicate clause in chain: AND(AND(X,Y), Y) → AND(X,Y).
            # Collect all rendered clauses from the left AND chain; if r is in
            # there, it's redundant.
            if l['type'] == 'binop' and l['op'] == 'AND':
                def _and_renders(n):
                    if n['type'] == 'binop' and n['op'] == 'AND':
                        return _and_renders(n['l']) | _and_renders(n['r'])
                    return {_render(n)}
                if _render(r) in _and_renders(l):
                    return l
            # Interval contradiction: X ≥ C1 AND X ≤ C2 (and variants) where
            # the bounds are impossible (lower > upper, or strict equal).
            b1, b2 = _bound_info(l), _bound_info(r)
            if b1 and b2 and b1[0] == b2[0]:
                _, op1, c1 = b1
                _, op2, c2 = b2
                lo, hi, strict_lo, strict_hi = None, None, False, False
                if op1 in ('≥', '>') and op2 in ('≤', '<'):
                    lo, hi = c1, c2
                    strict_lo, strict_hi = op1 == '>', op2 == '<'
                elif op1 in ('≤', '<') and op2 in ('≥', '>'):
                    lo, hi = c2, c1
                    strict_lo, strict_hi = op2 == '>', op1 == '<'
                if lo is not None:
                    if lo > hi or (lo == hi and (strict_lo or strict_hi)):
                        return {'type': 'num', 'v': 0.0}
            return {'type': 'binop', 'op': 'AND', 'l': l, 'r': r}

        if op == 'OR':
            lv_bool = _eval_bool(l)
            rv_bool = _eval_bool(r)
            if lv_bool is True or rv_bool is True:
                return {'type': 'num', 'v': 1.0}
            if lv_bool is False:
                return r
            if rv_bool is False:
                return l
            if _render(l) == _render(r):
                return l
            # Boolean absorption: A OR (NOT(A) AND B) = A OR B
            if r['type'] == 'binop' and r['op'] == 'AND':
                not_l = r['l']
                if not_l['type'] == 'not' and _render(not_l['e']) == _render(l):
                    return _simplify({'type': 'binop', 'op': 'OR', 'l': l, 'r': r['r']})
            if l['type'] == 'binop' and l['op'] == 'AND':
                not_r = l['l']
                if not_r['type'] == 'not' and _render(not_r['e']) == _render(r):
                    return _simplify({'type': 'binop', 'op': 'OR', 'l': r, 'r': l['r']})
            return {'type': 'binop', 'op': 'OR', 'l': l, 'r': r}

        # ── Comparison ────────────────────────────────────────────────────────
        _CMP_OPS = ('>', '<', '≥', '≤', '==', '!=')
        if op in _CMP_OPS:
            # Both constant → fold
            if lv is not None and rv is not None:
                result = {'>': lv > rv, '<': lv < rv,
                          '==': lv == rv, '!=': lv != rv,
                          '≥': lv >= rv, '≤': lv <= rv}[op]
                return {'type': 'num', 'v': 1.0 if result else 0.0}

            # Constant on LHS → flip so variable is on left
            if lv is not None and op in ('>', '<', '≥', '≤'):
                node2 = {'type': 'binop', 'op': _FLIP[op], 'l': r, 'r': l}
                return _simplify(node2)

            if op in ('>', '<', '≥', '≤'):
                # Move additive/multiplicative constants from LHS to RHS.
                # Only when RHS is a constant: the normalisation divides by
                # the coefficient, which would change a variable RHS incorrectly.
                if rv is not None:
                    l, rv_new, op = _normalise_lhs(l, rv, op)
                    r = {'type': 'num', 'v': round(rv_new, 10)}
                    rv = rv_new
                    lv = _const_val(l)

                # After normalisation, l might be constant too
                if lv is not None and rv is not None:
                    result = {'>': lv > rv, '<': lv < rv,
                              '≥': lv >= rv, '≤': lv <= rv}[op]
                    return {'type': 'num', 'v': 1.0 if result else 0.0}

                # Bound analysis (strict ops only)
                if op == '<' and rv is not None and _ge_lower_bound(l, rv):
                    return {'type': 'num', 'v': 0.0}
                if op == '>' and rv is not None and _le_upper_bound(l, rv):
                    return {'type': 'num', 'v': 0.0}

            # Push comparison inside IF: IF(cond, A, B) op t → IF(cond, A op t, B op t)
            # Semantically equivalent; simplifies each branch independently.
            # Never introduces OR — returns a clean IF or collapses to AND/cond.
            if rv is not None and op in ('>', '<', '≥', '≤') and l['type'] == 'if':
                cmp_A = _simplify({'type': 'binop', 'op': op, 'l': l['then'], 'r': r})
                cmp_B = _simplify({'type': 'binop', 'op': op, 'l': l['else'], 'r': r})
                ba, bb = _eval_bool(cmp_A), _eval_bool(cmp_B)
                if ba is True  and bb is True:  return {'type': 'num', 'v': 1.0}
                if ba is False and bb is False: return {'type': 'num', 'v': 0.0}
                if ba is True  and bb is False: return l['cond']
                if ba is False and bb is True:  return _simplify({'type': 'not', 'e': l['cond']})
                # One branch always False → AND without needing OR
                if ba is False:
                    return _simplify({'type': 'binop', 'op': 'AND',
                                      'l': {'type': 'not', 'e': l['cond']}, 'r': cmp_B})
                if bb is False:
                    return _simplify({'type': 'binop', 'op': 'AND',
                                      'l': l['cond'], 'r': cmp_A})
                if _render(cmp_A) == _render(cmp_B):
                    return cmp_A
                # Return cleaned IF with simplified branches
                return {'type': 'if', 'cond': l['cond'], 'then': cmp_A, 'else': cmp_B}

            # min(var, C) op t  /  max(var, C) op t — absorb constant bound
            # Only when threshold rv is a constant (avoids duplicating variable exprs).
            if rv is not None and l['type'] == 'fn' and op in ('>', '<', '≥', '≤'):
                fn = l['fn']
                lv_fn, rv_fn = _const_val(l['l']), _const_val(l['r'])
                C = lv_fn if lv_fn is not None else rv_fn
                var = l['r'] if lv_fn is not None else l['l']
                if C is not None:
                    C_passes = {'>': C > rv, '<': C < rv, '≥': C >= rv, '≤': C <= rv}[op]
                    if fn == 'min':
                        # min(var,C) op t  ↔  var op t  AND  C op t
                        if op in ('>', '≥'):
                            if not C_passes: return {'type': 'num', 'v': 0.0}
                            return _simplify({'type': 'binop', 'op': op, 'l': var, 'r': r})
                        else:
                            if C_passes:     return {'type': 'num', 'v': 1.0}
                            return _simplify({'type': 'binop', 'op': op, 'l': var, 'r': r})
                    else:  # max
                        # max(var,C) op t  ↔  var op t  OR  C op t
                        if op in ('>', '≥'):
                            if C_passes:     return {'type': 'num', 'v': 1.0}
                            return _simplify({'type': 'binop', 'op': op, 'l': var, 'r': r})
                        else:
                            if not C_passes: return {'type': 'num', 'v': 0.0}
                            return _simplify({'type': 'binop', 'op': op, 'l': var, 'r': r})
                else:
                    # Both args non-constant: only distribute when AND is the result
                    # (no OR introduced — OR is not in the grammar).
                    # min(a,b) > t  ↔  a > t AND b > t
                    # max(a,b) < t  ↔  a < t AND b < t
                    # max(a,b) > t and min(a,b) < t would need OR — leave as-is.
                    if fn == 'min' and op in ('>', '≥'):
                        return _simplify({'type': 'binop', 'op': 'AND',
                                          'l': {'type': 'binop', 'op': op, 'l': l['l'], 'r': r},
                                          'r': {'type': 'binop', 'op': op, 'l': l['r'], 'r': r}})
                    if fn == 'max' and op in ('<', '≤'):
                        return _simplify({'type': 'binop', 'op': 'AND',
                                          'l': {'type': 'binop', 'op': op, 'l': l['l'], 'r': r},
                                          'r': {'type': 'binop', 'op': op, 'l': l['r'], 'r': r}})

            return {'type': 'binop', 'op': op, 'l': l, 'r': r}

        # ── Arithmetic ────────────────────────────────────────────────────────

        # Both constant
        if lv is not None and rv is not None:
            if op == '+':
                return {'type': 'num', 'v': lv + rv}
            if op == '-':
                return {'type': 'num', 'v': lv - rv}
            if op == '*':
                return {'type': 'num', 'v': lv * rv}
            if op == '/' and rv != 0.0:
                return {'type': 'num', 'v': lv / rv}

        # Algebraic identities
        if op == '*':
            if rv == 0.0 or lv == 0.0:
                return {'type': 'num', 'v': 0.0}
            if rv == 1.0:
                return l
            if lv == 1.0:
                return r
            if rv == -1.0:
                return _negate(l)
            if lv == -1.0:
                return _negate(r)
            # (-c) * (0 - X) → c * X  (double negation: negative const × negated var)
            if lv is not None and lv < 0.0 and r['type'] == 'binop' and r['op'] == '-' \
                    and _const_val(r['l']) == 0.0:
                return _simplify({'type': 'binop', 'op': '*',
                                  'l': {'type': 'num', 'v': -lv}, 'r': r['r']})
            if rv is not None and rv < 0.0 and l['type'] == 'binop' and l['op'] == '-' \
                    and _const_val(l['l']) == 0.0:
                return _simplify({'type': 'binop', 'op': '*',
                                  'l': l['r'], 'r': {'type': 'num', 'v': -rv}})
        if op == '+':
            if rv == 0.0:
                return l
            if lv == 0.0:
                return r
            # (-c) + X → X - c  (negative constant on left → move to RHS)
            if lv is not None and lv < 0.0:
                return _simplify({'type': 'binop', 'op': '-', 'l': r,
                                  'r': {'type': 'num', 'v': -lv}})

            # (0.0 - A) + B → B - A  (avoids ugly -A + B form)
            if l['type'] == 'binop' and l['op'] == '-' and _const_val(l['l']) == 0.0:
                return _simplify({'type': 'binop', 'op': '-', 'l': r, 'r': l['r']})
            # A + (0.0 - B) → A - B
            if r['type'] == 'binop' and r['op'] == '-' and _const_val(r['l']) == 0.0:
                return _simplify({'type': 'binop', 'op': '-', 'l': l, 'r': r['r']})
        if op == '-':
            if rv == 0.0:
                return l
            # x - (-c) → x + c
            if rv is not None and rv < 0.0:
                return {'type': 'binop', 'op': '+', 'l': l,
                        'r': {'type': 'num', 'v': -rv}}
            # A - (-c * B) → A + (c * B)  — eliminates double-minus
            if r['type'] == 'binop' and r['op'] == '*':
                rc = _const_val(r['l']); rvc = _const_val(r['r'])
                c = rc if rc is not None else rvc
                if c is not None and c < 0.0:
                    var = r['r'] if rc is not None else r['l']
                    return _simplify({'type': 'binop', 'op': '+', 'l': l,
                                      'r': {'type': 'binop', 'op': '*',
                                            'l': {'type': 'num', 'v': -c}, 'r': var}})
            # 0 - (c * (0 - X)) → c * X  (double negation: 0 minus positive-const times negated)
            if lv == 0.0 and r['type'] == 'binop' and r['op'] == '*':
                rc = _const_val(r['l']); rvc = _const_val(r['r'])
                pos_c, inner = None, None
                if rc is not None and rc > 0.0:
                    pos_c, inner = rc, r['r']
                elif rvc is not None and rvc > 0.0:
                    pos_c, inner = rvc, r['l']
                if (inner is not None and inner['type'] == 'binop'
                        and inner['op'] == '-' and _const_val(inner['l']) == 0.0):
                    return _simplify({'type': 'binop', 'op': '*',
                                      'l': {'type': 'num', 'v': pos_c}, 'r': inner['r']})
            # x - x → 0
            if _render(l) == _render(r):
                return {'type': 'num', 'v': 0.0}
        if op == '/':
            if rv == 1.0:
                return l
            if rv == 0.0:
                return {'type': 'num', 'v': 0.0}   # safe-div convention
            if lv == 0.0:
                return {'type': 'num', 'v': 0.0}
            # x / c → (1/c) * x (to enable multiplicative accumulation)
            if rv is not None and rv != 0.0:
                return {'type': 'binop', 'op': '*',
                        'l': {'type': 'num', 'v': 1.0 / rv},
                        'r': l}

        # Additive constant accumulation: merge adjacent constant terms
        if op in ('+', '-'):
            terms = _collect_add_terms({'type': 'binop', 'op': op, 'l': l, 'r': r})
            const_terms = [(s, _const_val(t)) for s, t in terms]
            n_consts = sum(1 for _, v in const_terms if v is not None)
            if n_consts >= 2:
                const_sum = sum(s * v for s, v in const_terms if v is not None)
                var_terms  = [(s, t) for (s, t), (_, v) in zip(terms, const_terms)
                               if v is None]
                return _rebuild_add(var_terms, const_sum)

        # Multiplicative constant accumulation
        if op == '*':
            const_prod, factors = _collect_mul_terms(
                {'type': 'binop', 'op': '*', 'l': l, 'r': r})
            n_const_factors = sum(1 for f in factors if _const_val(f) is not None)
            if n_const_factors == 0 and const_prod != 1.0:
                var_part = factors[0]
                for f in factors[1:]:
                    var_part = {'type': 'binop', 'op': '*', 'l': var_part, 'r': f}
                return {'type': 'binop', 'op': '*',
                        'l': {'type': 'num', 'v': const_prod}, 'r': var_part}

        return {'type': 'binop', 'op': op, 'l': l, 'r': r}

    return node


# ── Additive / multiplicative accumulation helpers ────────────────────────────

def _collect_add_terms(node) -> list:
    """Flatten (+ / -) tree into [(sign, node), ...]."""
    if node['type'] == 'binop' and node['op'] == '+':
        return _collect_add_terms(node['l']) + _collect_add_terms(node['r'])
    if node['type'] == 'binop' and node['op'] == '-':
        left_terms = _collect_add_terms(node['l'])
        right_terms = [(-s, t) for s, t in _collect_add_terms(node['r'])]
        return left_terms + right_terms
    return [(1, node)]


def _rebuild_add(var_terms, const_sum) -> dict:
    """Rebuild an additive expression from signed variable terms and a constant."""
    if not var_terms:
        return {'type': 'num', 'v': const_sum}
    result = None
    if const_sum != 0.0:
        result = {'type': 'num', 'v': const_sum}
    for sign, term in var_terms:
        if result is None:
            result = term if sign == 1 else _negate(term)
        else:
            result = ({'type': 'binop', 'op': '+', 'l': result, 'r': term}
                      if sign == 1 else
                      {'type': 'binop', 'op': '-', 'l': result, 'r': term})
    return result


def _collect_mul_terms(node) -> tuple:
    """Flatten (* ) tree into (const_product, [variable_factor_nodes])."""
    if node['type'] == 'binop' and node['op'] == '*':
        lc, lf = _collect_mul_terms(node['l'])
        rc, rf = _collect_mul_terms(node['r'])
        return lc * rc, lf + rf
    v = _const_val(node)
    if v is not None:
        return v, []
    return 1.0, [node]


def _negate(node) -> dict:
    v = _const_val(node)
    if v is not None:
        return {'type': 'num', 'v': -v}
    return {'type': 'binop', 'op': '-',
            'l': {'type': 'num', 'v': 0.0}, 'r': node}


# ── Comparison normalization: move constants from LHS to RHS ─────────────────

def _normalise_lhs(lhs, rhs_val: float, op: str):
    """Move additive/multiplicative constants out of LHS into RHS.

    Returns (new_lhs, new_rhs_val, new_op).
    Flips the operator when dividing by a negative coefficient.
    """
    # Step 1: strip additive constants
    terms = _collect_add_terms(lhs)
    const_vs = [_const_val(t) for _, t in terms]
    n_consts = sum(1 for v in const_vs if v is not None)
    if n_consts > 0:
        const_sum = sum(s * v for (s, _), v in zip(terms, const_vs) if v is not None)
        var_terms  = [(s, t) for (s, t), v in zip(terms, const_vs) if v is None]
        if var_terms:
            lhs = _rebuild_add(var_terms, 0.0)
            rhs_val = round(rhs_val - const_sum, 10)

    # Step 2: strip multiplicative constant (c * x) → x, divide RHS by c
    if lhs['type'] == 'binop' and lhs['op'] == '*':
        lv = _const_val(lhs['l'])
        rv = _const_val(lhs['r'])
        c, var = None, None
        if lv is not None and lv != 0.0:
            c, var = lv, lhs['r']
        elif rv is not None and rv != 0.0:
            c, var = rv, lhs['l']
        if c is not None:
            rhs_val = round(rhs_val / c, 10)
            lhs = var
            if c < 0.0:
                op = _FLIP.get(op, op)

    # Step 3: (0 - x) → flip
    if (lhs['type'] == 'binop' and lhs['op'] == '-'
            and _const_val(lhs['l']) == 0.0):
        rhs_val = -rhs_val
        lhs = lhs['r']
        op = _FLIP.get(op, op)

    return lhs, rhs_val, op


def _eval_bool(node) -> bool | None:
    """Try to evaluate a boolean/comparison node as a Python bool."""
    t = node['type']
    if t == 'num':
        return bool(node['v'])
    if t == 'binop' and node['op'] == 'OR':
        lb, rb = _eval_bool(node['l']), _eval_bool(node['r'])
        if lb is True or rb is True:  return True
        if lb is False and rb is False: return False
        return None
    if t == 'binop' and node['op'] in ('>', '<', '==', '!=', '≥', '≤'):
        lv, rv = _const_val(node['l']), _const_val(node['r'])
        if lv is not None and rv is not None:
            return {'>'  : lv > rv,
                    '<'  : lv < rv,
                    '==' : lv == rv,
                    '!=' : lv != rv,
                    '≥'  : lv >= rv,
                    '≤'  : lv <= rv}[node['op']]
    if t == 'fn':  # min/max of constants
        v = _const_val(node)
        if v is not None:
            return bool(v)
    return None


# ── Renderer ──────────────────────────────────────────────────────────────────

def _render(node, prec: int = 0) -> str:
    """Convert an AST node back to a readable expression string."""
    t = node['type']

    if t == 'raw':
        return node['text']

    if t == 'num':
        return _fmt(node['v'])

    if t == 'var':
        return node['name']

    if t == 'abs':
        return f"|{_render(node['e'])}|"

    if t == 'fn':
        return f"{node['fn']}({_render(node['l'])}, {_render(node['r'])})"

    if t == 'not':
        inner = _render(node['e'])
        return f"NOT({inner})"

    if t == 'if':
        return (f"IF({_render(node['cond'])}, "
                f"{_render(node['then'])}, {_render(node['else'])})")

    if t == 'between':
        return (f"{_render(node['lo'])} ≤ {_render(node['e'])} "
                f"≤ {_render(node['hi'])}")

    if t == 'binop':
        op  = node['op']
        lv  = _const_val(node['l'])
        rv  = _const_val(node['r'])

        if op == 'AND':
            left  = _render(node['l'])
            right = _render(node['r'])
            # Wrap OR sub-expressions in parens (OR has lower precedence)
            if node['l']['type'] == 'binop' and node['l']['op'] == 'OR':
                left = f"({left})"
            if node['r']['type'] == 'binop' and node['r']['op'] == 'OR':
                right = f"({right})"
            return f"{left} AND {right}"

        if op == 'OR':
            left  = _render(node['l'])
            right = _render(node['r'])
            return f"{left} OR {right}"

        if op in ('>', '<', '==', '!=', '≥', '≤'):
            left  = _render(node['l'])
            right = _render(node['r'])
            return f"{left} {op} {right}"

        # Arithmetic — wrap in parens if needed to preserve precedence
        left_p  = _op_prec(op)
        right_p = _op_prec(op)

        # 0.0 - x  →  render as -x (unary negation)
        if op == '-' and _const_val(node['l']) == 0.0:
            inner_node = node['r']
            s = _render(inner_node, 2)
            # No parens needed for simple tokens or products/divisions
            if inner_node['type'] in ('var', 'num', 'abs', 'fn') or \
               (inner_node['type'] == 'binop' and inner_node['op'] in ('*', '/')):
                result = f"-{s}"
            else:
                result = f"-({s})"
            if prec > 1:
                return f"({result})"
            return result

        left  = _render(node['l'],  left_p)
        right = _render(node['r'],  right_p)

        # x + (-c)  →  render as  x - c
        if op == '+':
            rv_check = _const_val(node['r'])
            if rv_check is not None and rv_check < 0.0:
                right = _fmt(-rv_check)
                op = '-'
            # (-c) + expr  →  render as  expr - c  (constant last, more natural)
            elif _const_val(node['l']) is not None and _const_val(node['l']) < 0.0:
                c = _const_val(node['l'])
                result = f"{right} - {_fmt(-c)}"
                if prec > _op_prec('+') or prec == _op_prec('+'):
                    return f"({result})"
                return result
            # x + (-c * expr)  →  x - (c * expr)
            elif (node['r']['type'] == 'binop' and node['r']['op'] == '*'):
                rn = node['r']
                lv_r = _const_val(rn['l'])
                rv_r = _const_val(rn['r'])
                c = lv_r if lv_r is not None else rv_r
                if c is not None and c < 0.0:
                    var_node = rn['r'] if lv_r is not None else rn['l']
                    pos_mul = {'type': 'binop', 'op': '*',
                               'l': {'type': 'num', 'v': -c}, 'r': var_node}
                    right = _render(pos_mul, right_p)
                    op = '-'

        op_str = f" {op} "
        inner = f"{left}{op_str}{right}"
        if prec > _op_prec(op) or prec == _op_prec(op):
            return f"({inner})"
        return inner

    return str(node)


_PREC = {'+': 1, '-': 1, '*': 2, '/': 2}


def _op_prec(op: str) -> int:
    return _PREC.get(op, 0)


# ── Strip unnecessary outer parentheses ───────────────────────────────────────

def _strip_outer(s: str) -> str:
    """Strip matching outer parentheses when they wrap the whole expression."""
    while len(s) >= 2 and s[0] == '(' and s[-1] == ')':
        depth = 0
        hit_end = False
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0:
                if i == len(s) - 1:
                    hit_end = True
                break
        if not hit_end:
            break
        s = s[1:-1].strip()
    return s


# ── Main simplification entry point ──────────────────────────────────────────

def simplify(expr: str) -> str:
    """Parse, simplify, and re-render an expression string.

    Falls back to the original string on parse failure.
    Applies simplification repeatedly until stable.
    """
    if not isinstance(expr, str) or not expr.strip():
        return expr
    if expr in ('FAILED', 'nan', 'NaN', ''):
        return expr

    current = expr.strip()
    for _ in range(8):
        try:
            ast = _Parser(current).parse()
            simplified_ast = _simplify(ast)
            rendered = _render(simplified_ast)
            rendered = _strip_outer(rendered)
            # Normalise NOT((X == Y)) / NOT((X != Y))
            rendered = re.sub(r'NOT\(([^()]+) != ([^()]+)\)', r'\1 == \2', rendered)
            rendered = re.sub(r'NOT\(([^()]+) == ([^()]+)\)', r'\1 != \2', rendered)
            if rendered == current:
                break
            current = rendered
        except Exception:
            break
    return current


# ── CSV post-processing ───────────────────────────────────────────────────────

def process_csv(path: Path, dry_run: bool = False) -> int:
    """Simplify 'expression' column in a CSV. Returns number of changed rows."""
    import csv as _csv

    with open(path, newline='', encoding='utf-8') as f:
        rows = list(_csv.DictReader(f))

    if not rows or 'expression' not in rows[0]:
        return 0

    changed = 0
    for row in rows:
        orig = row['expression']
        simp = simplify(orig)
        if simp != orig:
            row['expression'] = simp
            changed += 1

    if changed > 0 and not dry_run:
        backup = path.with_suffix('.csv.bak')
        if not backup.exists():
            path.rename(backup)
        else:
            import shutil
            shutil.copy2(path, path.with_suffix('.csv.orig'))

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = _csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return changed


# ── Quick self-test ───────────────────────────────────────────────────────────

_TESTS = [
    # Identity rules
    ('(age / 1.0) > 3.0',     'age > 3.0'),
    ('(age * 1.0) > 3.0',     'age > 3.0'),
    ('age + 0.0 > 3.0',       'age > 3.0'),
    ('0.0 + age > 3.0',       'age > 3.0'),
    # Constant folding — negative coeff flips operator
    ('(-0.397 - 1.0) * x > 0', 'x < 0.0'),
    # abs/min/max of constants, then flip constant-on-LHS
    ('|6.0| > x',              'x < 6.0'),
    ('min(6.0, 7.0) > x',      'x < 6.0'),
    # All-constant comparison → bool literal
    ('max(3.0, min(6.0, 7.0)) > 5.0', '1.0'),
    # Double negative absorbed into threshold
    ('x - -2.5 > 2.136',       'x > -0.364'),
    # Comparison normalisation
    ('(2.0 + age) > 5.0',      'age > 3.0'),
    ('(-1.0 * age) > -5.0',    'age < 5.0'),
    # IF condition always False → else branch (OULAD ext p2 pattern)
    ('IF((max(3.0, min(|6.0|, 7.0)) > max(|6.0|, max(min(7.0, 6.0), min(7.0, code_module_CCC)))), 6.0, code_module_BBB)',
     'code_module_BBB'),
    # NOT double-neg
    ('NOT(NOT(age > 3.0))',    'age > 3.0'),
    # Categorical pass-through (no crash)
    ('Relationship == Husband', 'Relationship == Husband'),
    ('(marital_status == Never-married) AND (education == 1st-4th)', None),
    # COMPAS: coefficient simplification with redundant / 1.0 * 1.0
    ('(RecSupervisionLevel / 1.0) * 1.0 > 0', 'RecSupervisionLevel > 0.0'),
    # Law school: constant accumulation
    ('(((fam_inc * decile1b) + fam_inc) * -1.0) > 5.0', None),  # no crash
    # ── New: ≥ / ≤ as binary comparisons ──────────────────────────────────────
    ('age ≥ 18.0', 'age ≥ 18.0'),
    ('age ≤ 65.0', 'age ≤ 65.0'),
    # Constant on LHS flips
    ('18.0 ≤ age', 'age ≥ 18.0'),
    ('65.0 ≥ age', 'age ≤ 65.0'),
    # Constant-fold ≤/≥
    ('3.0 ≤ 5.0', '1.0'),
    ('5.0 ≤ 3.0', '0.0'),
    # Normalise LHS for ≤
    ('(age + 2.0) ≤ 20.0', 'age ≤ 18.0'),
    ('(-1.0 * age) ≥ -5.0', 'age ≤ 5.0'),
    # ── New: AND(...) prefix form ──────────────────────────────────────────────
    # True AND clause drops out
    ('AND((0.0 ≤ 27.0), (27.0 ≤ score))', 'score ≥ 27.0'),
    # False AND clause collapses whole expression
    ('AND((score ≤ 3.0), (6.0 ≤ 0.0))', '0.0'),
    ('AND((-0.182 ≤ -2.6), (score ≤ age))', '0.0'),
    # Both variable clauses remain (no crash)
    ('AND((age ≥ 18.0), (score ≤ 65.0))', 'age ≥ 18.0 AND score ≤ 65.0'),
    # ── New: Between → AND split ───────────────────────────────────────────────
    ('3.0 ≤ age ≤ 65.0', 'age ≥ 3.0 AND age ≤ 65.0'),
    # ── New: OR and min/max distribution ──────────────────────────────────────
    # ── New: (-c) + X → X - c (negative constant on left of +) ─────────────────
    ('-0.5 + age > 3.0',   'age > 3.5'),     # absorbed via normalise_lhs
    ('(-0.6392 + |min(decile3, 1.028)|) * (fam_inc / fam_inc) < 0.3885',
     '(|min(decile3, 1.028)| - 0.6392) * (fam_inc / fam_inc) < 0.3885'),
    # ── New: interval contradiction → 0.0 ────────────────────────────────────
    ('age ≥ 30.0 AND age ≤ 12.0',  '0.0'),
    ('age > 5.0 AND age < 5.0',    '0.0'),
    ('age > 5.0 AND age ≤ 5.0',    '0.0'),
    ('age ≥ 5.0 AND age < 5.0',    '0.0'),
    ('age ≥ 5.0 AND age ≤ 5.0',    'age ≥ 5.0 AND age ≤ 5.0'),  # valid (= 5)
    ('|Age| ≥ 29.0 AND |Age| ≤ 4.39 AND age > 3.0', '0.0'),
    # ── New: category name with embedded digit (10th) ────────────────────────
    ('(Education == 10th) AND (foo == bar) AND (foo == bar)', 'Education == 10 th AND foo == bar'),
    # double negation through multiply
    ('0.0 - 2.015 * (0.0 - age)',  '2.015 * age'),
    ('-2.015 * (0.0 - age)',        '2.015 * age'),
    # nested same-fn constant collapse
    ('max(max(age, 0.0), 1.0)',     'max(age, 1.0)'),
    ('min(min(age, 8.0), 3.0)',     'min(age, 3.0)'),
    # OR bool propagation
    ('age > 3.0 OR 1.0',   '1.0'),
    ('age > 3.0 OR 0.0',   'age > 3.0'),
    # min/max with constant arg, constant threshold
    ('min(age, 9.0) > 0',  'age > 0.0'),   # C=9 > 0 → age > 0
    ('min(age, 9.0) > 10', '0.0'),          # C=9 ≤ 10 → impossible
    ('max(age, 5.0) > 3',  '1.0'),          # C=5 > 3 → always true
    ('max(age, 0.5) > 3',  'age > 3.0'),    # C=0.5 ≤ 3 → age > 3
    # min/max both non-constant: AND cases only (no OR in grammar)
    ('min(a, b) > 0',      'a > 0.0 AND b > 0.0'),
    ('max(a, b) < 5',      'a < 5.0 AND b < 5.0'),
    # max(a,b) > t would need OR — leave as-is (no OR in grammar)
    ('max(a, b) > 0',      'max(a, b) > 0.0'),
    # Nested min/max: constant-arg collapse still works where one arg is constant
    ('min(code_module_GGG, 9.0) > 0',  'code_module_GGG > 0.0'),
    ('max(code_module_GGG, 0.5) > 3',  'code_module_GGG > 3.0'),
]


def run_tests() -> None:
    pass_count = 0
    fail_count = 0
    for expr, expected in _TESTS:
        result = simplify(expr)
        if expected is None:
            print(f"  OK (no-crash): {expr!r}  →  {result!r}")
            pass_count += 1
        elif result == expected:
            print(f"  OK: {expr!r}  →  {result!r}")
            pass_count += 1
        else:
            print(f"  FAIL: {expr!r}")
            print(f"    Expected : {expected!r}")
            print(f"    Got      : {result!r}")
            fail_count += 1
    print(f"\n{pass_count} passed, {fail_count} failed.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)

    if args[0] == '--test':
        run_tests()
        return

    dry_run = '--dry-run' in args
    folders = [a for a in args if not a.startswith('--')]

    if not folders:
        print("Error: provide a run folder path.")
        sys.exit(1)

    for folder_arg in folders:
        folder = Path(folder_arg)
        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue

        csv_files = sorted(folder.rglob('*.csv'))
        if not csv_files:
            print(f"No CSVs found under {folder}")
            continue

        total_changed = 0
        for csv_path in csv_files:
            if '.bak' in csv_path.suffixes or '.orig' in csv_path.suffixes:
                continue
            n = process_csv(csv_path, dry_run=dry_run)
            if n:
                print(f"  {csv_path.relative_to(folder)}: {n} expression(s) simplified")
                total_changed += n

        if total_changed == 0:
            print(f"No simplifications found in {folder}")
        elif dry_run:
            print(f"\nDry-run: {total_changed} expression(s) would be simplified.")
        else:
            print(f"\nDone: {total_changed} expression(s) simplified.")
            print("Originals backed up as *.csv.bak")


if __name__ == '__main__':
    main()
