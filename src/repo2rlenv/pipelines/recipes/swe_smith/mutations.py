# [MODULE DOC] Formatting-preserving single-site procedural mutations (LibCST), inspired by SWE-
# smith's operations/control_flow strategies: operator flips, condition inversion and integer
# constant changes inside functions, enumerated deterministically.
"""Formatting-preserving procedural mutation strategies inspired by SWE-smith.

Source: SWE-bench/SWE-smith, MIT, 9b74ac08118a85c39c356802f7961893af73e07f,
bug_gen/procedural/python/{operations,control_flow}.py. Owned single-site
enumeration replaces upstream probabilistic multi-edit sampling. LibCST is an
ordinary parsing library; no upstream research package is imported.
"""

from __future__ import annotations

import difflib
import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Mutation:
    # [CLASS DOC] Frozen dataclass for one mutation: id (20-hex sha256 prefix of filename+mutated
    # source), operator (flip_operator / invert_condition / change_constant), entity (dotted
    # class/function scope), line (1-based line of the mutated node), source (full mutated text),
    # patch (unified diff a/ -> b/).
    id: str
    operator: str
    entity: str
    line: int
    source: str
    patch: str


def generate_mutations(
    source: str, filename: str, *, seed: int = 0, limit: int = 100
) -> list[Mutation]:
    # [FUNCTION DOC] Enumerate valid mutations of a Python file.
    # Inputs: source, filename (validated as a safe relative asset path), keyword seed (shuffle
    # seed, default 0), limit (max results, default 100; must be >=1 else ValueError). Collects
    # sites only within function bodies: flips of binary/boolean/comparison operators per a fixed
    # table, wrapping if-tests in a parenthesized "not", and integer literals +/-1 (|value| <= 1e6).
    # Sites are shuffled by seed; each is applied by identity-replacing the node, discarding no-op
    # or unparsable results and duplicates. Output: list of Mutation, at most limit.
    # Raises ValueError (limit, unsafe filename) and LibCST parse errors for invalid input source.
    """Enumerate source-valid edits inside functions, retaining surrounding text."""
    import libcst as cst
    from libcst.metadata import MetadataWrapper, PositionProvider

    from repo2rlenv.emitter.bundle import relative_asset_path

    relative_asset_path("environment/" + filename)
    if limit < 1:
        raise ValueError("Mutation limit must be positive")
    wrapper = MetadataWrapper(cst.parse_module(source))
    positions = wrapper.resolve(PositionProvider)
    sites: list[tuple[cst.CSTNode, cst.CSTNode, str, str]] = []
    flips = {
        cst.Add: cst.Subtract,
        cst.Subtract: cst.Add,
        cst.Multiply: cst.Divide,
        cst.Divide: cst.Multiply,
        cst.FloorDivide: cst.Modulo,
        cst.Modulo: cst.FloorDivide,
        cst.And: cst.Or,
        cst.Or: cst.And,
        cst.Equal: cst.NotEqual,
        cst.NotEqual: cst.Equal,
        cst.LessThan: cst.GreaterThan,
        cst.GreaterThan: cst.LessThan,
        cst.LessThanEqual: cst.GreaterThanEqual,
        cst.GreaterThanEqual: cst.LessThanEqual,
        cst.In: cst.NotIn,
        cst.NotIn: cst.In,
        cst.Is: cst.IsNot,
        cst.IsNot: cst.Is,
    }

    class Collect(cst.CSTVisitor):
        # [CLASS DOC] HELPER: LibCST visitor tracking class/function scope names and function depth,
        # appending (original, replacement, operator, entity) tuples to the enclosing sites list for
        # nodes inside functions. visit_BooleanOperation and visit_ComparisonTarget alias the
        # binary-operation visitor.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Initialise an empty scope stack and function_depth 0.
            self.scope: list[str] = []
            self.function_depth = 0

        def visit_ClassDef(self, node):
            # [FUNCTION DOC] HELPER: Push the class name onto scope.
            self.scope.append(node.name.value)

        def leave_ClassDef(self, node):
            # [FUNCTION DOC] HELPER: Pop the class name from scope.
            self.scope.pop()

        def visit_FunctionDef(self, node):
            # [FUNCTION DOC] HELPER: Push the function name and increment function_depth.
            self.scope.append(node.name.value)
            self.function_depth += 1

        def leave_FunctionDef(self, node):
            # [FUNCTION DOC] HELPER: Pop scope and decrement function_depth.
            self.scope.pop()
            self.function_depth -= 1

        def add(self, original, replacement, operator):
            # [FUNCTION DOC] HELPER: Record a site (with dotted scope as entity) only when inside a
            # function.
            if self.function_depth:
                sites.append((original, replacement, operator, ".".join(self.scope)))

        def visit_BinaryOperation(self, node):
            # [FUNCTION DOC] HELPER: If the node's operator type has a flip, record a flip_operator
            # site with a freshly constructed flipped operator (default whitespace).
            replacement = flips.get(type(node.operator))
            if replacement:
                self.add(node, node.with_changes(operator=replacement()), "flip_operator")

        visit_BooleanOperation = visit_BinaryOperation
        visit_ComparisonTarget = visit_BinaryOperation

        def visit_If(self, node):
            # [FUNCTION DOC] HELPER: Record an invert_condition site replacing the test with
            # not(parenthesized test).
            self.add(
                node,
                node.with_changes(
                    test=cst.UnaryOperation(
                        cst.Not(),
                        cst.ensure_type(node.test, cst.BaseExpression).with_changes(
                            lpar=[cst.LeftParen()], rpar=[cst.RightParen()]
                        ),
                    )
                ),
                "invert_condition",
            )

        def visit_Integer(self, node):
            # [FUNCTION DOC] HELPER: For integer literals with |value| <= 1,000,000, record
            # change_constant sites for value-1 and value+1.
            value = node.evaluated_value
            if abs(value) <= 1_000_000:
                for delta in (-1, 1):
                    self.add(node, cst.parse_expression(str(value + delta)), "change_constant")

    wrapper.module.visit(Collect())
    random.Random(seed).shuffle(sites)
    result: list[Mutation] = []
    seen: set[str] = set()

    class Replace(cst.CSTTransformer):
        # [CLASS DOC] HELPER: LibCST transformer that substitutes one specific node (by identity)
        # with its replacement.
        def __init__(self, original, replacement):
            # [FUNCTION DOC] HELPER: Store the original node and replacement.
            self.original, self.replacement = original, replacement

        def on_leave(self, original_node, updated_node):
            # [FUNCTION DOC] HELPER: Return the replacement when leaving the original node,
            # otherwise the updated node.
            return self.replacement if original_node is self.original else updated_node

    for original, replacement, operator, entity in sites:
        mutated = wrapper.module.visit(Replace(original, replacement)).code
        if mutated == source:
            continue
        # Parse only: generated task code is never executed on the controller.
        try:
            cst.parse_module(mutated)
        except cst.ParserSyntaxError:
            continue
        identity = hashlib.sha256((filename + "\0" + mutated).encode()).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        patch = "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                mutated.splitlines(keepends=True),
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
            )
        )
        result.append(
            Mutation(
                identity[:20], operator, entity, positions[original].start.line, mutated, patch
            )
        )
        if len(result) >= limit:
            break
    return result
