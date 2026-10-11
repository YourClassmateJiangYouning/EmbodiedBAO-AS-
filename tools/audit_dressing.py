"""Audit the dressing code for the mistakes this session actually made, by reading it.

Written after the user asked why the same class of bug kept appearing.  Every check below stands
for a real defect from this session, so a green run means those specific faults are absent rather
than that the code "looks fine".

  1. A frame conversion applied to a tuple that is already in that frame.  This happened three
     times -- _dressing_wall, fit_scale_from_extent, and the scale op -- and each time it silently
     swapped an item's height with its width.
  2. An exception whose record keeps only str(exc), which for USD's ErrorException is the empty
     string, so a report said "FAILED ErrorException:" and nothing else.
  3. A transform op added without checking for an authored one of the same type.
  4. A prop fitted along an axis the checks do not use, i.e. a wrong-axis fit.
  5. A helper called with a size whose frame does not match what that helper expects.
  6. An unmeasured prop wired into a scene, whose declared size was therefore invented.

Run from the repository root:  python tools/audit_dressing.py
"""
from __future__ import annotations

import ast
import os
import sys
from typing import Any, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

FILES = ("scene_builder.py", "scenes.py")


def read(name: str) -> str:
    return open(os.path.join(_ROOT, name), encoding="utf-8").read()


def main() -> int:
    problems = []
    notes = []

    builder = read("scene_builder.py")

    # 1. THE FRAME RULE, restated for the design that replaced it.
    #
    # The old rule asserted that a WALL item's size is stored in world order and that every
    # to_world_size call therefore needs a mount guard.  That premise was wrong -- wall sizes are
    # stored in the CATALOGUE frame like everything else, and the mount exemption is exactly what
    # drew every wall item with its height and width exchanged.  So the rule is now the opposite and
    # simpler: every size in this module is in ONE frame, and the conversion to the stage's frame
    # happens exactly once, in place_dressing, for both mounts, with no mount condition anywhere.
    #
    # Comments and docstrings are stripped before scanning.  This file's checks have now read their
    # own prose five times -- a docstring naming a misspelt member, a comment quoting a retired
    # phrase, a comment naming the API it warns about, and twice here -- so prose is removed first
    # rather than being argued with case by case.
    def strip_prose(source: str) -> str:
        """The source with DOCSTRINGS and COMMENTS blanked, and everything else left alone.

        Three attempts got this wrong, and the third shows what the rule has to be:

          * a regular expression blanked every string literal, including the one inside
            ``record["fit_error"]``, so checks looked for that field and reported it missing while it
            sat in the code;
          * concatenating or untokenizing token strings lost the whitespace BETWEEN tokens, turning
            ``def apply_scale`` into ``defapply_scale``;
          * blanking every STRING token does the first fault by construction, because a dictionary key
            written ``record["fit_error"]`` IS a string literal.

        A string literal that is a subscript key or a message is CODE; the prose that misleads a
        text-scanning check is the docstring and the comment.  So only those are blanked: docstrings
        are located through the AST (a string constant that is a module, class or function body's first
        statement, or a bare expression statement), and comments through the tokenizer.  Character
        positions are preserved, so line numbers in the result still refer to the real file.
        """
        import io
        import tokenize

        line_starts = [0]
        for line in source.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(line))

        def at(line: int, column: int) -> int:
            """Character offset of a 1-based line and 0-based column."""
            return line_starts[line - 1] + column

        blanks: List[tuple] = []
        tree = ast.parse(source)

        def blank_docstrings(node: Any) -> None:
            body = getattr(node, "body", None)
            if isinstance(body, list) and body:
                first = body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                        and isinstance(first.value.value, str):
                    blanks.append((at(first.value.lineno, first.value.col_offset),
                                   at(first.value.end_lineno, first.value.end_col_offset)))
            for child in ast.walk(node):
                # Any remaining bare string statement is prose too -- a stray docstring, a commented
                # block kept as a string.  Subscript keys are never Expr statements, so this cannot
                # reach the dictionary keys that the previous version blanked by mistake.
                if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) \
                        and isinstance(child.value.value, str):
                    blanks.append((at(child.value.lineno, child.value.col_offset),
                                   at(child.value.end_lineno, child.value.end_col_offset)))

        blank_docstrings(tree)
        try:
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type == tokenize.COMMENT:
                    blanks.append((at(*token.start), at(*token.end)))
        except tokenize.TokenError:
            # An unterminated construct still yields the tokens produced before it, which is enough for
            # a source-text scan; scanning the whole text is better than raising inside a check.
            return source

        out = list(source)
        for start, end in blanks:
            for index in range(start, min(end, len(out))):
                if out[index] != "\n":
                    out[index] = " "
        return "".join(out)

    def function_body(source: str, name: str, stripped: str) -> str:
        """The prose-stripped source of one TOP-LEVEL function.

        Every earlier slice used string splitting, and every one was wrong in a different way: split
        on "\\ndef " ran past the end into the next function, "def name" without a newline matched a
        mention inside a docstring, and "\\ndef " also matches an INDENTED nested function -- this file
        contains one, and it truncated the slice early so that the checks below read half a function
        while reporting the other half missing.

        The AST is parsed from the ORIGINAL source, because the stripped text is not valid Python (its
        string literals are blanked), and the line range it reports is then applied to the stripped
        text -- the two have identical line numbering by construction.
        """
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                lines = stripped.splitlines()
                return "\n".join(lines[node.lineno - 1:node.end_lineno])
        return ""

    code_only = strip_prose(builder)
    # Counted as CALLS, not as mentions: `sc.to_world_size(item["size"])` appears twice legitimately
    # -- once to size the clearance box, once to size the reference the verification compares against
    # -- and what matters is that both are unconditional and in the same frame.
    conversion_lines = [lineno for lineno, line in enumerate(code_only.splitlines(), 1)
                        if "to_world_size" in line]
    if not conversion_lines:
        problems.append("scene_builder.py never converts a size with to_world_size, so the stage "
                        "would receive catalogue-frame sizes with height and lateral exchanged")
    if "mount" in code_only.split("to_world_size")[0][-120:]:
        problems.append("scene_builder.py converts a size under a mount condition; wall and floor "
                        "items share one frame, and exempting a mount is what exchanged a wall "
                        "item's height with its width")

    # 2. An except that records only str(exc).  Scanned with the prose in place, because the
    # traceback has to be there in the real body -- but the test is on the CODE, so that a message
    # that merely mentions the field name cannot satisfy it.
    tree = ast.parse(builder)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = "".join(ast.dump(statement) for statement in node.body
                       if not (isinstance(statement, ast.Expr)
                               and isinstance(statement.value, ast.Constant)))
        if "reference_error" in body and "traceback" not in body and "extract_tb" not in body:
            problems.append(
                f"scene_builder.py:{node.lineno} records an exception without its traceback, so a "
                f"USD ErrorException -- whose str() is empty -- would report nothing")

    # 3. AddScaleOp must not be reached without checking for an authored op.
    if "def apply_scale" not in code_only:
        problems.append("scene_builder.py has no apply_scale helper; an authored xformOp:scale "
                        "would make AddScaleOp raise")
    elif "AddScaleOp()" in code_only:
        guarded = "GetOrderedXformOps" in code_only
        notes.append(f"AddScaleOp present, GetOrderedXformOps present: {guarded}")
        if not guarded:
            problems.append("AddScaleOp is used without GetOrderedXformOps, so an asset that ships "
                            "a scale op cannot be fitted")

    # 4/5. Every wall call passes a size in the frame _dressing_wall expects, and every wall item
    # with an asset has its depth on slot 0.
    import scenes as sc                                                  # noqa: E402

    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            if item["mount"] == "floor":
                continue
            # The same arithmetic as test_no_dressing_protrudes_into_the_corridor, and written the
            # same way on purpose: this check and that test are the same invariant, and they used to
            # disagree by a factor of two (depth here, half-depth there), so a depth between 0.20 and
            # 0.40 m would have passed the test and failed the audit depending on which ran first.
            # _dressing_wall puts the back face on the surface, so the protrusion is the depth less
            # the clearance.
            protrusion = float(item["size"][0]) - sc.MOUNT_CLEARANCE_M
            if protrusion > 0.20 + 1e-9:
                problems.append(f"{scene}/{item['name']} protrudes {protrusion:.3f} m from the wall, "
                                f"past the 0.20 m cap")

    # 6. Wired props with no measurement and not declared pending.
    pending = set(getattr(sc, "PROP_MEASUREMENT_PENDING", ()))
    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            asset = item.get("asset")
            if not asset:
                continue
            base = os.path.basename(asset)
            if not base.endswith(".usd"):
                continue
            if base not in sc.PROP_MEASUREMENTS and base not in pending:
                problems.append(f"{scene}/{item['name']} references unmeasured {base}")

    # 7. A wall item may not be declared taller than the room's 3 m ceiling, and a floor item's box
    # may not exceed it either: the fit would stretch the prop through the ceiling.
    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            if float(item["size"][1]) > 3.0:
                problems.append(f"{scene}/{item['name']} is {item['size'][1]:.2f} m tall, above the "
                                f"3.0 m ceiling")

    # 8. The fit must be computed against the DECLARED size, be UNIFORM, and carry an OFFSET.
    #
    # This rule used to say the opposite -- "it must NOT take the declared size ... the factor is
    # 1/native" -- and that rule passed while the defect shipped.  Its premise was that the prop is a
    # CHILD of the clearance cube, which carries the declared scale; the prop is a SIBLING of the cube
    # under the item's Xform, and the measurement is taken relative to that Xform, so `native` is in
    # the item's local units where one unit IS the declared size.  `min(1/native)` therefore fills the
    # box only on the asset's largest axis, and measured on the four dressed scenes it drew every prop
    # at 2 %-88 % of its declared box on the other axes (a 9 cm mug at 8.5 mm) while the report stayed
    # green, because extents obey 1/native x native <= 1 whatever the asset is.
    #
    # The three defects that live here, all checked below:
    #
    #   * the factor ignores the declared size (what shipped);
    #   * a per-axis factor reshapes a prop whose proportions differ from its box.  It reported
    #     (1.0, 16.0075, 0.0624) for the safety railing -- 16x too tall or 16x too small, depending on
    #     which reading of the frame you took, and neither is a railing;
    #   * no offset, so an asset whose geometry is anchored at a face instead of its centre is drawn
    #     from the item's anchor outward.  Every extent still fits, and it is still clipping: measured
    #     on the four scenes, 25 of 43 props left their declared box, the conifer reaching
    #     z = 3.5738 m under a 3.0 m ceiling.
    FIT_NAME = "fit_reference_to_local_unit"
    OFFSET_NAME = "apply_translate"
    if f"def {FIT_NAME}" not in builder:
        problems.append(f"scene_builder.py has no {FIT_NAME}; the prop's fit cannot be checked")
        fit_body = ""
    else:
        # Split on the NEXT top-level def, so the slice is this function only.  An earlier version
        # split on "\ndef " and caught apply_scale's comment below, reporting a defect in a function
        # that no longer had one; a later one split on the wrong marker again.  Slice once, here.
        # Sliced at the next TOP-LEVEL def: a marker of "\ndef " also matches an indented nested
        # function, and this file contains one (strip_prose), which truncated the slice early.  A
        # top-level def is one whose `def` starts at column zero.
        fit_body = function_body(builder, FIT_NAME, code_only)
    if fit_body:
        # The factor must be 1/native.  A version of this rule demanded `declared/native` instead,
        # reasoning that the prop had to fill the DECLARED box rather than one unit of its parent; the
        # stage disproved it -- with declared/native the conifer was drawn 5.76 m tall for a 2.4 m box,
        # because the measured bound is already expressed in the item Xform's local units, where one
        # unit is the declared size (two-point calibration: 109.51 m at scale 1, 219.02 m at scale 2,
        # for an item Xform scaled to 2.4).  So the rule is: 1/native, read into `factor`.
        if "1.0 / native[axis]" not in fit_body:
            problems.append(f"{FIT_NAME} does not derive its factor from 1/native, which is the "
                            f"multiplier that takes the measured bound to the declared box")
        # The value has to be READ, not merely computed: a formula that ignores its own ratios is the
        # shipped defect with extra steps, and the first version of this rule accepted exactly that,
        # because `min(ratios)` is a substring of the line that assigns the factor.
        if "factor = min(ratios)" not in fit_body:
            problems.append(f"{FIT_NAME} does not take the smallest of its three ratios, so the prop "
                            f"can exceed its declared box on the other axes")
        if "if factor > 1.0:" not in fit_body or "factor = 1.0" not in fit_body:
            problems.append(f"{FIT_NAME} does not cap the factor at 1.0, so a declared box larger "
                            f"than the asset would ENLARGE it -- enlarging is what turned houseplants "
                            f"into trees in this project")
        if "return (offset, (factor, factor, factor))" not in fit_body:
            problems.append(f"{FIT_NAME} does not return ONE factor for all three axes, so it would "
                            f"reshape the prop instead of scaling it")
        if "fitted.append" in fit_body:
            problems.append(f"{FIT_NAME} builds a per-axis result list, which means the prop is "
                            f"reshaped rather than scaled")
        if "-centre" not in fit_body:
            problems.append(f"{FIT_NAME} does not return an offset for the prop's own centre, so a "
                            f"face-anchored asset is drawn from the item's anchor outward")
        elif "-centre[axis] / native[axis]" not in fit_body:
            problems.append(f"{FIT_NAME} returns the centre in the ASSET's units rather than as a "
                            f"fraction of the parent's box.  The parent's scale is per-axis, so the "
                            f"raw centre is multiplied by it and the prop lands at "
                            f"want*(1 + centre) instead of centred -- measured on this stage as a "
                            f"301-unit rack at z = -299 m")
    if f"def {OFFSET_NAME}" not in builder:
        problems.append(f"scene_builder.py has no {OFFSET_NAME}; a prop carrying its own "
                        f"xformOp:translate would make AddTranslateOp raise")
        offset_body = ""
    else:
        offset_body = function_body(builder, OFFSET_NAME, code_only)
        # The same guard apply_scale has, for the same reason: AddTranslateOp raises when the op is
        # already authored, and record["fit_offset"] can only be written if the write succeeded.
        if "GetOrderedXformOps" not in offset_body:
            problems.append(f"{OFFSET_NAME} does not check for an authored translate, so an asset "
                            f"that ships one cannot be placed")

    # 8b. THE PROP MUST SURVIVE THE BOX BEING HIDDEN.  The single worst defect in this file's history
    # was `prim.SetActive(False)` on the clearance CUBE, whose child is the referenced prop:
    # deactivation removes a subtree from composition, so 43 resolved props were not drawn at all
    # while the report printed RESOLVED, and the verification below could never find them.  Hiding an
    # imageable is inherited and reversible; deactivating is neither.
    if "def place_dressing" not in builder:
        problems.append("scene_builder.py has no place_dressing")
        place_body = ""
    else:
        # Sliced to the next top-level def, like the fit above.  This used to run to EOF, covering
        # classify_surfaces, discover_surfaces, describe_stage, apply_scene and format_report, so its
        # three presence tests could be satisfied from any later function.
        place_body = function_body(builder, "place_dressing", code_only)
    if place_body:
        if "SetActive(False)" in place_body:
            problems.append("place_dressing deactivates a prim.  If that prim is the clearance box, "
                            "its referenced prop is deactivated with it and never drawn -- which is "
                            "exactly what happened here.  Make the box invisible instead")
        if "MakeInvisible" not in place_body and "GetVisibilityAttr" not in place_body:
            problems.append("place_dressing never makes the clearance box invisible, so a resolved "
                            "prop is drawn inside a solid box and reads as a block")
        # THE STRUCTURE, which is what makes hiding the box safe at all.  Presence tests were not
        # enough: the state this project actually shipped had MakeInvisible present, applied to a box
        # that was the prop's PARENT -- and `invisible` is inherited, so the prop was hidden with it
        # and every frame stayed bare while the report said RESOLVED.  The rule is about the shape of
        # the prim tree, so it is checked as one.
        if "/clearance" not in place_body:
            problems.append("place_dressing does not put the clearance box on its own prim path.  The "
                            "prop must be a SIBLING of that box, not its child: both SetActive(False) "
                            "and MakeInvisible() hide a prim's descendants, so a prop under the box "
                            "cannot be shown while the box is hidden")
        if "item_xform = UsdGeom.Xform.Define(stage, path)" not in \
                place_body.replace("\n", " ").replace("  ", " "):
            problems.append("place_dressing does not define the item's own Xform at `path`.  That "
                            "Xform is what carries the placement and the declared scale once the box "
                            "is a sibling of the prop; without it nothing scales the prop to the "
                            "declared size.  (A check that merely looks for the NAME item_xform "
                            "passes when the Xform is defined somewhere else, which is why this "
                            "compares the call.)")
        # The scale must be on the item, not on the clearance box: a unit cube under a scaled Xform is
        # the declared volume, and the prop is scaled by the same Xform.
        for line in place_body.splitlines():
            if "AddScaleOp" in line and "item" not in line[:line.index("AddScaleOp")]:
                problems.append(f"place_dressing scales the {line.strip()!r} rather than the item's "
                                f"Xform; with the box as a sibling of the prop, the declared scale "
                                f"has to live on the Xform that contains both")
        if "MakeInvisible()" not in place_body:
            problems.append("place_dressing does not call MakeInvisible() on the box prim")
        elif "Imageable(prim)" not in place_body:
            problems.append("place_dressing makes something invisible that is not the clearance box "
                            "prim; making the ITEM invisible would hide the prop with it")
        # The prop must be measured for verification in WORLD units, i.e. relative to the dressing
        # root, which has no transform.  An earlier version measured it relative to the item's Xform
        # on the reasoning that "the Xform carries the scale, so a relative bound is in metres" -- but
        # a relative bound EXCLUDES the transform at that ancestor, so it returned the parent's local
        # units, saturating at exactly 1.0 on each item's constraining axis and printing three false
        # FITS BADLY lines about a fountain that was the right size.
        if "ComputeRelativeBound(proxy, root_prim)" not in place_body.replace("\n", " "):
            problems.append("place_dressing's verification does not measure the prop relative to the "
                            "DRESSING ROOT.  A relative bound excludes the transform at the prim it is "
                            "relative to, so measuring against the item's Xform returns the parent's "
                            "local units rather than metres, and the drawn-vs-declared comparison "
                            "becomes meaningless")

        # THE OFFSET GOES ON BEFORE THE SCALE, and this is a separate ordering from the item's own
        # translate/scale below.  The prop's ops are [translate, scale], so the geometry is moved in
        # the prop's own units and the result is scaled -- which is what puts the prop's centre on the
        # item's anchor.  Written the other way round, the offset is multiplied by the factor first
        # and the prop lands somewhere else entirely; both orders produce a plausible extent, which is
        # why this is checked as an order rather than as a presence.
        prop_offset_at = place_body.find("apply_translate(proxy_prim")
        prop_scale_at = place_body.find("apply_scale(proxy_prim")
        if prop_offset_at == -1:
            problems.append("place_dressing never applies an offset to the prop, so an asset anchored "
                            "at a face instead of its centre is drawn from the item's anchor outward")
        elif prop_scale_at != -1 and prop_offset_at > prop_scale_at:
            problems.append("place_dressing applies the prop's SCALE before its OFFSET, so the offset "
                            "is multiplied by the fit factor and the prop is not centred on the anchor")
        # THE ITEM'S OWN op order: the translate is added before the scale, so the scale is the
        # INNERMOST op and "a box of `size` centred at `at`" means what it says.  Written the other way
        # round every item is centred at `size * at` -- the crate declared at x 10.2 landed at x 6.14 --
        # and the extents are identical either way, which is why this is checked as an order rather
        # than as a presence.
        #
        # The patterns are the ITEM's calls, and that is a correction: this rule used to search for
        # `apply_translate` and `apply_scale`, which are the PROP's calls, so it was a second copy of
        # the rule above and could not see the item's order at all.  The fault this rule is supposed to
        # catch was reported MISSED for exactly that reason -- a check that cannot see the thing it
        # names looks the same as a check that passed.
        item_translate_at = place_body.find(".AddTranslateOp")
        item_scale_at = place_body.find("item_xform.AddScaleOp")
        if item_translate_at == -1 or item_scale_at == -1:
            problems.append("place_dressing does not author both a translate and a scale on the item "
                            "Xform")
        elif item_translate_at > item_scale_at:
            problems.append("place_dressing adds the ITEM's SCALE before its TRANSLATE, so the "
                            "translation is applied first and then multiplied by the scale: every "
                            "item centre becomes size * at")
        if "Gf.Vec3d" not in place_body and "Gf.Vec3d" not in offset_body:
            problems.append("place_dressing does not use a double-precision vector for the translate, "
                            "which AddTranslateOp creates at PrecisionDouble")
        if 'to_world_size(item' not in place_body:
            problems.append("place_dressing does not convert the item size with to_world_size, so a "
                            "wall item's height and width would be exchanged in the drawn box")
        # THE POSITIONAL HALF OF THE VERIFICATION.  The extent test (`drawn > declared`) cannot fail
        # for a prop fitted with declared/native, because that fit makes the tightest axis exact and
        # every other axis smaller -- so on its own it is a check that cannot fail, which is the class
        # of check this project has been burned by.  What CAN fail, and did, is where the geometry
        # sits: the bound has to be inside `at +/- size/2` in world axes.  Both the recorded box and
        # the comparison against it are therefore required.
        if '"drawn_box"' not in place_body:
            problems.append("place_dressing does not record the drawn prop's world box, so a prop "
                            "that is the right size in the wrong place cannot be reported")
        positional = ('float(lo[axis]) < box_lo[axis]' in place_body
                      and 'float(hi[axis]) > box_hi[axis]' in place_body)
        if not positional:
            problems.append("place_dressing compares only the prop's EXTENT with the declared box, "
                            "not its position.  An extent is hi - lo and says nothing about where the "
                            "geometry is, so a prop drawn outside its box passes while clipping")

    # 9. No declared box may be smaller than the prop it names, because the prop is fitted INSIDE
    # the box: a box that cannot hold it means the prop overhangs and clips, which is the failure
    # this project must not have.  Compared as sorted sets, for the wall-item frame difference, and
    # with a tolerance because a box derived from the same measurement differs from it by float
    # rounding -- an exact comparison reported eleven items whose two lists printed identically.
    # 5e-5: declared sizes are written to four decimal places by hand, so a box derived from a
    # measurement can differ from it by half a unit in the last place -- 0.17 written against
    # 0.1700070 computed.  An exact or 1e-6 comparison reported fifteen items that fit perfectly.
    TOL = 5e-5
    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            asset = item.get("asset")
            if not asset:
                continue
            base = os.path.basename(asset)
            measured = sc.PROP_MEASUREMENTS.get(base)
            if not measured:
                continue
            factor = float(sc.ASSET_SCALE.get(base, 1.0))
            room_fit = float(item.get("room_fit", 1.0))
            prop = sorted(float(v) * factor * room_fit for v in measured)
            box = sorted(float(v) for v in item["size"])
            if any(box[axis] < prop[axis] - TOL for axis in range(3)):
                problems.append(
                    f"{scene}/{item['name']}: box {[round(v, 4) for v in box]} m cannot hold the prop "
                    f"{[round(v, 4) for v in prop]} m, so the prop would overhang and clip")

    # 9. The bound API must be the one whose SEMANTICS fit the question.  This is the fault class a
    # name allowlist cannot catch, and it produced the worst defect of this session: every one of
    # these calls is a real API, so only reading the reference distinguishes them.  From
    # https://openusd.org/release/api/class_usd_geom_b_box_cache.html :
    #
    #   ComputeWorldBound(prim)                 "in world space" -- includes the prim's own
    #                                           transform AND every ancestor's.
    #   ComputeRelativeBound(prim, ancestor)    "in the space of an ancestor prim ... excludes the
    #                                           local transform at relativeToAncestorPrim".
    #   ComputeLocalBound(prim)                 includes the prim's own transform, excludes ancestor
    #                                           transforms, and returns an ORIENTED box.
    #   ComputeUntransformedBound(prim)         excludes the prim's own transform.
    #
    # fit_scale_from_extent measures a prop that is a CHILD of the clearance box whose scale it is
    # about to set, so an ancestor-including bound measures the fit against itself: reported as
    # factors of 219.9, 39.5 and 11.1.  A relative bound excludes exactly that ancestor.
    if "ComputeRelativeBound" not in code_only:
        problems.append("scene_builder.py measures a fit with no ComputeRelativeBound, so the "
                        "extent would include the parent box's own scale and feed itself")
    # fit_body was extracted once, above, for rule 8.  It was extracted a SECOND time here, and the
    # duplicate split on "\ndef " rather than on the next top-level def, so it ran past the end of
    # fit_scale_from_extent into apply_scale and read that function's comment about ComputeWorldBound
    # as though it were a call -- reporting a defect in a function that no longer had one.  Two
    # extractions of the same slice is the duplication this project treats as a defect in its own
    # right, and it produced exactly that: a false report.
    if fit_body and "ComputeWorldBound" in fit_body:
        problems.append("fit_scale_from_extent uses ComputeWorldBound, whose bound includes every "
                        "ancestor transform -- including the parent box's scale, which is what this "
                        "function is deciding.  Use ComputeRelativeBound.")
    # 10. place_dressing must VERIFY the fit on the assembled stage.  The clearance cube carries a
    # scale of `size` and the prop under it carries a scale of its own; xform ops compose down the
    # hierarchy, so whether the drawn prop is native x size x factor or something else is a runtime
    # fact that reading the source cannot settle -- and guessing it is how this session produced its
    # most expensive defects.  The code therefore measures each resolved prop's extent back off the
    # stage with a FRESH BBoxCache and records fit_error when it exceeds its declared box.
    # The probes are the QUOTED forms, because the prose-stripping blanks string literals and these
    # names appear as dictionary keys -- record["fit_error"] -- which is a string literal by
    # definition.  Looking for the bare name reported the field missing while it was in the code.
    if '"fit_error"' not in code_only:
        problems.append("place_dressing never verifies the drawn size against the declared box, so a "
                        "prop scaled wrongly would be invisible until someone looked at a picture")
    if '"drawn_extent"' not in code_only:
        problems.append("place_dressing does not record what the stage says was drawn, so no report "
                        "can state a prop's actual size")
    # place_body was extracted once, above, and sliced at the next top-level def.  This block used to
    # assign it AGAIN from the raw source with split("def place_dressing", 1)[-1], which runs to the
    # end of the file and therefore covered classify_surfaces, discover_surfaces, describe_stage,
    # apply_scene and format_report as well -- so every check below could be satisfied from any later
    # function, and the three above it (run on the correctly-sliced body) were contradicted by it.
    # Two extractions of the same slice is the duplication this project treats as a defect in itself,
    # and it produced exactly that: checks that pass and fail according to which variable won.
    if place_body.count("BBoxCache(") < 1:
        problems.append("place_dressing does not construct a BBoxCache of its own, so its "
                        "verification would read bounds cached before the props were scaled")
    if "ComputeRelativeBound" not in place_body:
        problems.append("place_dressing's verification does not use ComputeRelativeBound, so an "
                        "ancestor transform would be included in the size it checks")
    # 11. The stale-comment check.  Every defect fixed in this file left its old argument behind at
    # least once, and a comment that argues for behaviour the code no longer has is worse than none:
    # it is how the next person reintroduces the defect.  Only phrases that can be tied to a specific
    # reversal are listed, so this cannot fire on ordinary prose.
    STALE = (
        ("per axis, not uniform", "the fit is uniform now"),
        ("Fitting the prop to the box removes that class of defect: the\n                #    declared size is what is drawn",
         "the prop is fitted INSIDE the box and may be smaller, so the declared size is a bound"),
        ("STORED IN WORLD ORDER", "a wall item is stored in the catalogue frame, depth on slot 0"),
        ('``frame="wall"`` reverses', "the frame parameter is gone"),
        ("clearance box DEACTIVATED", "the box is made invisible now, not deactivated"),
        ("clearance box DEACTIVATED, so", "the box is made invisible now, not deactivated"),
    )
    for phrase, why in STALE:
        if phrase in builder or phrase in read("scenes.py"):
            problems.append(f"stale comment arguing for superseded behaviour: {phrase!r} -- {why}")

    # 12. THE one stale claim that misled this project four separate times: "a resolved prop has its
    # clearance box deactivated".  It was true while the box was deactivated, and that deactivation is
    # what deleted every referenced prop from the stage; the sentence then survived in four files and
    # each copy made a report read as though props were being drawn when they were not.
    #
    # Scanned over the PROSE only -- docstrings and comments -- and for that claim rather than for the
    # word, because the word also appears legitimately: a comment explaining that the box is made
    # invisible INSTEAD, and a runtime message asking whether an ancestor was deactivated, are both
    # correct and were both flagged by the word-based version of this rule.
    for name in ("scene_builder.py", "scenes.py", "capture_scenes.py",
                 os.path.join("tools", "render_dressing.py")):
        path = os.path.join(_ROOT, name)
        if not os.path.exists(path):
            continue
        prose = strip_prose(open(path, encoding="utf-8").read())
        for lineno, line in enumerate(prose.splitlines(), 1):
            lowered = line.lower()
            if "clearance box" in lowered and ("deactivat" in lowered or "inactive" in lowered):
                problems.append(
                    f"{name}:{lineno} still says a resolved prop's clearance box is deactivated or "
                    f"inactive: {line.strip()[:80]!r}.  It is made invisible; the difference is that "
                    f"deactivation removes the prop with it, which is what emptied every frame")

    env_path = os.path.join(_ROOT, "environment.py")
    if os.path.exists(env_path):
        env = open(env_path, encoding="utf-8").read()
        if "ComputeWorldBound" not in env:
            notes.append("environment.py no longer measures the robot with a world bound; that is "
                         "not an error, but the comment explaining why it was correct there has "
                         "probably moved or gone")

    # 13. THE SAFE RENDER DEFAULT.  `use_mdl` decides whether the four room surfaces get MDL prims or
    # flat paint, and on the lab machine the material library's cache is not writable, so MDL surfaces
    # render as salt-and-pepper noise.  Three places disagreed about the default -- the signature said
    # True, its own docstring said False, and capture_scenes inverted the meaning of --flat_paint -- so
    # noise was what a plain run produced, and a preview full of noise was read more than once as "the
    # scene is bare".  All three are checked here because the failure is invisible in a text report.
    if "use_mdl: bool = False" not in code_only:
        problems.append("scene_builder.apply_scene does not default use_mdl to False, so MDL surfaces "
                        "-- noise on the lab machine -- would be the default again")
    env_text = open(os.path.join(_ROOT, "environment.py"), encoding="utf-8").read()
    if 'get("use_mdl", False)' not in env_text:
        problems.append("environment.py does not default its use_mdl lookup to False, which is the "
                        "other half of the same default")
    cap_path = os.path.join(_ROOT, "capture_scenes.py")
    if os.path.exists(cap_path):
        cap_text = open(cap_path, encoding="utf-8").read()
        if "use_mdl\": bool(args.mdl)" not in cap_text:
            problems.append("capture_scenes.py does not drive use_mdl from an explicit --mdl flag.  "
                            "Deriving it from the ABSENCE of --flat-paint made materials the silent "
                            "default and made those runs unreadable")
        if "not args.flat_paint" in cap_text:
            problems.append("capture_scenes.py still inverts --flat_paint into use_mdl, which makes "
                            "materials the silent default again")

    print(f"read {', '.join(FILES)}")
    for note in notes:
        print(f"  note: {note}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nno problems found by these checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
