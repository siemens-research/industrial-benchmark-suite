"""Utility functions to inspect variables."""

import sys
import inspect
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def inspect_variable(var, indent=0):
    """Recursively inspect a variable and print its type and structure."""
    var_type = type(var)
    indent_str = " " * indent
    len_str = ""
    if isinstance(var, (dict, list, tuple, set)):
        len_str = f" ({len(var)} elements)"
    elif isinstance(var, (np.ndarray, pd.Series, pd.DataFrame)):
        len_str = f" (shape {var.shape})"

    print(f"{indent_str}Type: {var_type}{len_str}")

    if isinstance(var, dict):
        for k in var.keys():
            print(f"{indent_str}{k}:")
            inspect_variable(var[k], indent=indent + 4)
    if isinstance(var, (list, tuple)):
        for i, elem in enumerate(var):
            print(f"{indent_str}{i}:")
            inspect_variable(elem, indent=indent + 4)


def summarize_variable(
    value,
    *,
    name=None,
    as_str=False,
    with_methods=False,
    with_fields=True,
    private_methods=False,
    private_fields=False,
    num_elements=10,
    num_higher_elements=1,
    indent=4,
):
    """Summarize key information about ``value``.

    Parameters
    ----------
    value : any
        Object to summarize.
    name : str, optional
        Optional label to prefix the summary.
    as_str : bool, default False
        When True, return the summary string instead of printing.
    with_methods : bool, default False
        Include non-dunder methods in the summary.
    with_fields : bool, default True
        Include non-dunder, non-callable attributes in the summary.
    private_methods : bool, default False
        Include private (single-underscore) methods.
    private_fields : bool, default False
        Include private (single-underscore) fields.
    num_elements : int, default 10
        How many elements to show before trimming with ellipses.
    num_higher_elements : int, default 1
        For arrays with ndim > 2, how many outer slices to render.
    indent : int, default 4
        Number of spaces to indent nested values.
    """
    lines = []
    seen = set()

    def _indent(level):
        return " " * (indent * level)

    def _head_tail_indices(length):
        if length <= num_elements:
            return list(range(length)), [], False
        head = max(1, num_elements // 2)
        tail = max(1, num_elements - head)
        return list(range(head)), list(range(length - tail, length)), True

    def _is_complex(val):
        return isinstance(val, (Mapping, np.ndarray, pd.Series, pd.DataFrame)) or (
            isinstance(val, Sequence) and not isinstance(val, (str, bytes))
        )

    def _format_methods(obj, level):
        methods = []
        for attr in dir(obj):
            if attr.startswith("__") and attr.endswith("__"):
                continue
            if attr.startswith("_") and not private_methods:
                continue
            try:
                val = getattr(obj, attr)
            except Exception:
                continue
            if callable(val):
                methods.append(attr)
        head, tail, trimmed = _head_tail_indices(len(methods))
        display = [methods[i] for i in head] + (["..."] if trimmed else []) + [methods[i] for i in tail]
        if display:
            lines.append(f"{_indent(level)}Methods: {', '.join(display)}")

    def _format_fields(obj, level):
        fields = []
        for attr in dir(obj):
            if attr.startswith("__") and attr.endswith("__"):
                continue
            if attr.startswith("_") and not private_fields:
                continue
            try:
                val = getattr(obj, attr)
            except Exception:
                continue
            if callable(val):
                continue
            fields.append((attr, val))
        head, tail, trimmed = _head_tail_indices(len(fields))
        display = [fields[i] for i in head] + ([('...', '...')] if trimmed else []) + [fields[i] for i in tail]
        if display:
            lines.append(f"{_indent(level)}Fields:")
            for attr, val in display:
                if attr == "...":
                    lines.append(f"{_indent(level + 1)}...")
                elif _is_complex(val):
                    lines.append(f"{_indent(level + 1)}{attr}:")
                    _format(val, level + 2)
                else:
                    lines.append(f"{_indent(level + 1)}{attr}: {val}")

    def _format_array(arr, level):
        arr_np = np.asarray(arr)
        lines.append(f"{_indent(level)}Array: shape={arr_np.shape}, dtype={arr_np.dtype}")
        if arr_np.ndim == 0:
            lines.append(f"{_indent(level + 1)}{arr_np.item()}")
            return

        def _format_1d(one_d, lvl):
            head, tail, trimmed = _head_tail_indices(len(one_d))
            display = [one_d[i] for i in head] + (["..."] if trimmed else []) + [one_d[i] for i in tail]
            lines.append(f"{_indent(lvl)}[{', '.join(str(v) for v in display)}]")

        def _format_2d(two_d, lvl):
            row_head, row_tail, row_trimmed = _head_tail_indices(two_d.shape[0])
            col_head, col_tail, col_trimmed = _head_tail_indices(two_d.shape[1])
            row_indices = row_head + (["..."] if row_trimmed else []) + row_tail
            col_indices = col_head + (["..."] if col_trimmed else []) + col_tail

            for r_idx in row_indices:
                if r_idx == "...":
                    lines.append(f"{_indent(lvl)}...")
                    continue
                values = []
                for c_idx in col_indices:
                    if c_idx == "...":
                        values.append("...")
                    else:
                        values.append(str(two_d[r_idx, c_idx]))
                lines.append(f"{_indent(lvl)}[{r_idx}]: {', '.join(values)}")

        if arr_np.ndim == 1:
            _format_1d(arr_np.tolist(), level + 1)
        elif arr_np.ndim == 2:
            _format_2d(arr_np, level + 1)
        else:
            outer = arr_np.reshape(-1, *arr_np.shape[-2:])
            shown = min(num_higher_elements, outer.shape[0]) if num_higher_elements else 0
            for idx in range(shown):
                lines.append(f"{_indent(level + 1)}[outer {idx}]:")
                _format_2d(outer[idx], level + 2)
            if outer.shape[0] > shown:
                lines.append(f"{_indent(level + 1)}:::")

    def _format_series(series, level):
        series_name = f" (name: {series.name})" if getattr(series, "name", None) is not None else ""
        lines.append(f"{_indent(level)}Type: pd.Series{series_name}")
        items = list(series.items())
        head, tail, trimmed = _head_tail_indices(len(items))
        display = [items[i] for i in head] + (["..."] if trimmed else []) + [items[i] for i in tail]
        for entry in display:
            if entry == "...":
                lines.append(f"{_indent(level + 1)}...")
            else:
                idx, val = entry
                lines.append(f"{_indent(level + 1)}{idx}: {val}")

    def _format_dataframe(df, level):
        lines.append(f"{_indent(level)}Type: pd.DataFrame shape={df.shape}")
        cols = list(df.columns)
        c_head, c_tail, c_trimmed = _head_tail_indices(len(cols))
        shown_cols = [cols[i] for i in c_head] + (["..."] if c_trimmed else []) + [cols[i] for i in c_tail]
        lines.append(f"{_indent(level + 1)}Columns: {', '.join(str(c) for c in shown_cols)}")

        row_indices = list(df.index)
        r_head, r_tail, r_trimmed = _head_tail_indices(len(row_indices))
        display_rows = [row_indices[i] for i in r_head] + (["..."] if r_trimmed else []) + [row_indices[i] for i in r_tail]

        for idx in display_rows:
            if idx == "...":
                lines.append(f"{_indent(level + 1)}...")
                continue
            row = df.loc[idx]
            parts = []
            for col in shown_cols:
                if col == "...":
                    parts.append("...")
                else:
                    parts.append(f"{col}: {row[col]}")
            lines.append(f"{_indent(level + 1)}{idx} | {', '.join(parts)}")

    def _format(val, level):
        if id(val) in seen:
            lines.append(f"{_indent(level)}<recursion>")
            return
        seen.add(id(val))

        type_label = f"{type(val)}"
        if name and level == 0:
            lines.append(f"{_indent(level)}{name}: {type_label}")
        else:
            lines.append(f"{_indent(level)}Type: {type_label}")

        if with_methods:
            _format_methods(val, level + 1)
        if with_fields:
            _format_fields(val, level + 1)

        if isinstance(val, Mapping):
            items = list(val.items())
            head, tail, trimmed = _head_tail_indices(len(items))
            display = [items[i] for i in head] + (["..."] if trimmed else []) + [items[i] for i in tail]
            for entry in display:
                if entry == "...":
                    lines.append(f"{_indent(level + 1)}...")
                    continue
                key, sub_val = entry
                lines.append(f"{_indent(level + 1)}{key}:")
                _format(sub_val, level + 2)
        elif isinstance(val, (str, bytes)):
            lines.append(f"{_indent(level + 1)}{val}")
        elif isinstance(val, np.ndarray):
            _format_array(val, level + 1)
        elif isinstance(val, pd.Series):
            _format_series(val, level + 1)
        elif isinstance(val, pd.DataFrame):
            _format_dataframe(val, level + 1)
        elif isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            seq_vals = list(val)
            head, tail, trimmed = _head_tail_indices(len(seq_vals))
            indices = head + (["..."] if trimmed else []) + tail
            for idx in indices:
                if idx == "...":
                    lines.append(f"{_indent(level + 1)}...")
                    continue
                lines.append(f"{_indent(level + 1)}{idx}:")
                _format(seq_vals[idx], level + 2)
        else:
            lines.append(f"{_indent(level + 1)}{val}")

    _format(value, 0)

    output = "\n".join(lines)
    if as_str:
        return output
    print(output)
    return None


def gather_exceptions():
    """Collect all loaded exception classes that inherit from BaseException."""
    excs = set()
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, BaseException):
                excs.add(obj)
    return excs


def qualified_name(cls):
    """Return module-qualified class name."""
    return f"{cls.__module__}.{cls.__name__}"


def build_tree(classes):
    """Build parent -> children mapping for the given exception classes."""
    tree = defaultdict(list)
    for cls in classes:
        for base in cls.__mro__[1:]:  # skip self
            if base in classes:
                tree[base].append(cls)
                break
    return tree


def print_tree(tree, root, indent=0, visited=None):
    """Pretty-print the inheritance tree starting from root."""
    if visited is None:
        visited = set()
    if root in visited:
        return
    visited.add(root)
    print("  " * indent + qualified_name(root))
    for child in sorted(tree.get(root, []), key=lambda c: qualified_name(c)):
        print_tree(tree, child, indent + 1, visited)


def show_exception_hierarchy():
    """Gather and print the exception class hierarchy.
    
    >>> from demo_dashboard.utils.inspect import show_exception_hierarchy
    >>> show_exception_hierarchy()
    """
    exceptions = gather_exceptions()
    tree = build_tree(exceptions)

    # Start from BaseException to show full tree
    print_tree(tree, BaseException)
