"""Static check for names that no enclosing scope provides.

Several bugs in this project were the same shape: a name was referenced but
nothing bound it.  Two real examples, both of which only fired on a real Isaac
Sim install after a ~150 s startup:

* `main.py` referenced `BAOExperimentRunner` inside a module-level helper after
  the import had been (correctly) moved inside a function.
* `environment.py` referenced Isaac Sim names that a deferred import no longer
  provided.

A per-function unit test cannot see either one, so this walks the file as a
scope tree: each function/class/module scope collects the names it binds
(imports, defs, assignments, for/with/try targets, arguments, walrus, nested
comprehension targets) and every name load is resolved against that scope and
its enclosing scopes.  Unresolved names are reported.

Deliberately conservative in one direction: names bound anywhere inside a
scope count for the whole scope, so a use-before-assignment is not reported.
Only genuinely unbound names are, which is the class of bug that hurt us.

    python tools/check_names.py
"""

from __future__ import annotations

import ast
import builtins
import os
from typing import Dict, List, Optional, Set

MODULES = [
    "main.py",
    "environment.py",
    "protocol.py",
    "experiments.py",
    "ai_agent.py",
    "analysis.py",
    "persistence.py",
    "test_bao_geometry.py",
    "test_bao_integration.py",
    "test_bao_persistence.py",
]

# Interpreter-provided module globals.
IMPLICIT_GLOBALS: Set[str] = {
    "__file__",
    "__name__",
    "__doc__",
    "__spec__",
    "__package__",
    "__loader__",
    "__builtins__",
    "__annotations__",
    "__cached__",
}

# Names that only exist on machines with Isaac Sim, bound by deferred imports.
# Keep this list short and justified; it must never hide a real bug.
ALLOWED_UNDEFINED: Set[str] = set()

BUILTINS: Set[str] = set(dir(builtins))


class Scope:
    """One lexical scope: the names it binds plus its parent."""

    def __init__(self, kind: str, parent: Optional["Scope"] = None) -> None:
        self.kind = kind
        self.parent = parent
        self.bound: Set[str] = set()
        self.globals_declared: Set[str] = set()
        self.nonlocals_declared: Set[str] = set()

    def resolves(self, name: str) -> bool:
        if name in self.bound:
            return True
        if name in BUILTINS or name in IMPLICIT_GLOBALS or name in ALLOWED_UNDEFINED:
            return True
        scope: Optional[Scope] = self
        while scope is not None:
            if name in scope.bound:
                return True
            if name in scope.globals_declared or name in scope.nonlocals_declared:
                # Declared to live elsewhere; accept rather than guess.
                return True
            scope = scope.parent
        return False


def bind_target(target: ast.AST, scope: Scope) -> None:
    """Record every name bound by an assignment target."""
    if isinstance(target, ast.Name):
        scope.bound.add(target.id)
    elif isinstance(target, ast.Starred):
        bind_target(target.value, scope)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            bind_target(element, scope)
    # Attribute / Subscript targets bind nothing new.


def bind_function_signature(node: ast.AST, scope: Scope) -> None:
    args = node.args  # type: ignore[attr-defined]
    for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
        scope.bound.add(arg.arg)
    if args.vararg:
        scope.bound.add(args.vararg.arg)
    if args.kwarg:
        scope.bound.add(args.kwarg.arg)


def collect_bindings(body: List[ast.stmt], scope: Scope) -> None:
    """Collect names bound directly by ``body`` into ``scope``.

    Nested functions and classes get their *own* scope, so their bodies are
    deliberately NOT descended into here -- otherwise an import inside a
    function would incorrectly appear to bind a module-level name, which is
    precisely the bug this checker exists to catch.  Only the function name
    (and for a class, the class name) is bound in the enclosing scope.
    """
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                scope.bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.bound.add(stmt.name)
        elif isinstance(stmt, ast.ClassDef):
            scope.bound.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                bind_target(target, scope)
                for sub in ast.walk(target):
                    if isinstance(sub, ast.NamedExpr):
                        bind_target(sub.target, scope)
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
            bind_target(stmt.target, scope)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            bind_target(stmt.target, scope)
            collect_bindings(stmt.body, scope)
            collect_bindings(stmt.orelse, scope)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None:
                    bind_target(item.optional_vars, scope)
            collect_bindings(stmt.body, scope)
        elif isinstance(stmt, ast.Try):
            collect_bindings(stmt.body, scope)
            collect_bindings(stmt.orelse, scope)
            collect_bindings(stmt.finalbody, scope)
            for handler in stmt.handlers:
                if handler.name:
                    scope.bound.add(handler.name)
                collect_bindings(handler.body, scope)
        elif isinstance(stmt, (ast.If, ast.While)):
            collect_bindings(stmt.body, scope)
            collect_bindings(stmt.orelse, scope)
        elif isinstance(stmt, ast.Global):
            scope.globals_declared.update(stmt.names)
        elif isinstance(stmt, ast.Nonlocal):
            scope.nonlocals_declared.update(stmt.names)

    # Walrus and comprehension targets, but only within this scope's own
    # statements (not inside nested function bodies).
    for node in iter_nodes_shallow_scopes(ast.Module(body=list(body), type_ignores=[])):
        if isinstance(node, ast.NamedExpr):
            bind_target(node.target, scope)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in node.generators:
                bind_target(gen.target, scope)


def iter_nodes_shallow_scopes(root: ast.AST):
    """Walk ``root`` but stop at nested function/class/lambda boundaries.

    Their bodies belong to their own scope; their decorators and default
    values are evaluated in the enclosing scope and are visited separately by
    the caller.
    """
    stack = [root]
    first = True
    while stack:
        node = stack.pop()
        if not first and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        first = False
        yield node
        stack.extend(ast.iter_child_nodes(node))


class Checker:
    def __init__(self, path: str) -> None:
        self.path = path
        self.problems: List[str] = []

    def check_scope(self, body: List[ast.stmt], scope: Scope) -> None:
        for stmt in body:
            self.check_stmt(stmt, scope)

    def check_stmt(self, stmt: ast.stmt, scope: Scope) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Decorators and defaults are evaluated in the ENCLOSING scope.
            for decorator in stmt.decorator_list:
                self.check_expr(decorator, scope)
            for default in list(stmt.args.defaults) + [
                d for d in stmt.args.kw_defaults if d is not None
            ]:
                self.check_expr(default, scope)
            if stmt.returns is not None:
                self.check_expr(stmt.returns, scope)

            inner = Scope("function", scope)
            bind_function_signature(stmt, inner)
            collect_bindings(stmt.body, inner)
            self.check_scope(stmt.body, inner)
            return

        if isinstance(stmt, ast.ClassDef):
            for decorator in stmt.decorator_list:
                self.check_expr(decorator, scope)
            for base in stmt.bases:
                self.check_expr(base, scope)
            inner = Scope("class", scope)
            collect_bindings(stmt.body, inner)
            self.check_scope(stmt.body, inner)
            return

        # Everything else: check loaded names in this subtree, but do not
        # descend into nested function/class scopes (handled above).
        for node in iter_nodes_shallow_scopes(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if not scope.resolves(node.id):
                    self.problems.append(
                        f"{self.path}:{node.lineno}: '{node.id}' is used but no "
                        f"enclosing scope binds it"
                    )

        # Recurse into nested blocks with the same scope.
        for field, value in ast.iter_fields(stmt):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.stmt):
                        self.check_stmt(item, scope)
                    elif isinstance(item, ast.excepthandler):
                        self.check_scope(item.body, scope)
            elif isinstance(value, ast.stmt):
                self.check_stmt(value, scope)

    def check_expr(self, expr: ast.AST, scope: Scope) -> None:
        for node in iter_nodes_shallow_scopes(expr):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if not scope.resolves(node.id):
                    self.problems.append(
                        f"{self.path}:{node.lineno}: '{node.id}' is used but no "
                        f"enclosing scope binds it"
                    )


def check_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source, filename=path)

    module_scope = Scope("module")
    collect_bindings(tree.body, module_scope)

    checker = Checker(path)
    checker.check_scope(tree.body, module_scope)

    seen: Set[str] = set()
    unique: List[str] = []
    for problem in checker.problems:
        if problem not in seen:
            seen.add(problem)
            unique.append(problem)
    return unique


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    problems: List[str] = []
    checked = 0
    for module in MODULES:
        path = os.path.join(root, module)
        if not os.path.exists(path):
            continue
        checked += 1
        problems.extend(check_file(path))

    if problems:
        print("Unbound-name problems found:\n")
        for problem in problems:
            print("  " + problem)
        print(f"\n{len(problems)} problem(s) in {checked} files")
        return 1
    print(f"no unbound names in {checked} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
