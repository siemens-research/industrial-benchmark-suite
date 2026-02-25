"""
Collection of utility functions to process lists and dictionaries.

@author: Holger Schoener <holger.schoener@siemens.com>
"""

import copy
import math
import re
import warnings
from collections import OrderedDict
from itertools import filterfalse
from itertools import groupby
from operator import itemgetter
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd


def is_sequence(arg):
    """Indicate whether arg is a sequence, but not a string."""
    if arg is None:
        return False
    return (
        not hasattr(arg, "strip")
        and not isinstance(arg, np.generic)
        and (hasattr(arg, "__getitem__") or hasattr(arg, "__iter__"))
    )


def flatten(x, levels=None, do_not_flatten=(str, dict)):
    """
    Merge nested list levels into the outermose level.
    
    Returns a single list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables), up to the given level. If level is None, then recurse
    until no iterable is left, otherwise at most perform the given nubmer
    of recursion levels (e.g. level == 1 only flattens one level of inner
    lists). Recursion is stopped if no iterable (containing __iter__) is
    found, or if the iterable belongs to one of the types given in
    do_not_flatten.

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]
    """
    result = []
    for elem in x:
        #if isinstance(elem, (list, tuple)):
        if (not levels or levels > 0) and hasattr(elem, "__iter__") and not isinstance(elem, do_not_flatten):
            result.extend(flatten(elem, levels=(levels-1 if levels else None), do_not_flatten=do_not_flatten))
        else:
            result.append(elem)
    return result


def first(items, pred):
    """Return first element which matches the condition `pred` (a callable receiving one element each call)."""
    return next((i for i in items if pred(i)), None)


def argsort(seq):
    """Return the indices list, which when used as indexer into `seq` would make it ordered."""
    return sorted(range(len(seq)), key=seq.__getitem__)


def split_string(string, sep=","):
    """Split the given string at all occurences of sep, and trim the resulting substrings of white space."""
    if isinstance(sep, str):
        return map(lambda x: x.strip(), string.split(sep))
    return map(lambda x: x.strip(), re.split(sep, string))


def exclude(lst, elem):
    """Return a copy of list lst, with all instances of elem removed."""
    res = copy.copy(lst)
    res.remove(elem)
    return res


def find_first_index(pred, lst):
    """Return the index of the first element for which pred(element) evaluates to True; returns None if no such element exists."""
    for i, elem in enumerate(lst):
        if pred(elem):
            return i
    return None


def coalesce(*args, ret_index=False):
    """
    Return the first argument which is not None (or its index in the argument list and the element itself, if `ret_index`).
    
    Returns `None` (or `(None, None)`) in case all arguments are `None`.
    """
    for i, elem in enumerate(args):
        if elem is not None:
            if ret_index:
                return i, elem
            return elem
    if ret_index:
        return None, None
    return None


def partition(lst: Sequence, props: Sequence):
    """
    Partition a list lst into a list of mutually exclusive sublists.
         
    The returned sublists together cover the original list `lst`. The split is performed according to the unique
    values of `props`. `props` is assumed to be a list with as many elements as lst.
    """
    assert len(lst) == len(props)
    # if all([ isinstance(p, dict) for p in props ]):
    #     unique_props = map(dict, set(tuple(x.items()) for x in props))
    # else:
    unique_props = list(set(props))
    res = [[] for p in unique_props]
    for i, elem in enumerate(lst):
        part_num = unique_props.index(props[i])
        res[part_num].append(elem)
    return res


def partition_lengths(
    n: int,
    k: Optional[int] = None,
    m: Optional[int] = None,  # pylint: disable=invalid-name
    result: Union[str, Sequence] = "lengths",
) -> Union[int, List]:
    """
    Give properties (e.g. number, sizes) of splitting `n` elements into almost equally sized partitions.
    
    Computes the lengths of partitions if partitioning n elemenst into either m partitions, or into partitions of maximal
    size k (with actual sizes being as large as possible), with sizes of each partition differing by at most 1.
    :param n: length of list to partition
    :param k: maximal size of subpartitions, either k or m have to be provided, but not both
    :param m: number of subpartitions, either k or m have to be provided, but not both
    :param result: one of "lengths" (default), "max_length", "min_length", "num_max_length", "num_min_length", "number",
           "indices", or a list of some of these strings. The corresponding information is returned (as a single element,
           or as a list in the order specified by these strings)
    :return: a single value or a list according to the 'result' argument
    """
    if (k is None and m is None) or (k is not None and m is not None):
        raise ValueError("exactly one of 'k' and 'm' have to be provided (not being None), but k={}, m={}".format(k, m))
    if m is None:
        m, _ = divmod(n+k-1, k)     # type: ignore      # k is not None, otherwise m would not be None
    else: # if k is None:
        k, _ = divmod(n+m-1, m)
    
    assert isinstance(m, int)
    assert isinstance(k, int)

    length, x = divmod(n, m)
    inds = [ length*i + min(i, x) for i in range(m+1) ]
    lens = [ inds[i+1]-inds[i] for i in range(m) ]
    x = x if x>0 else m
    y = m-x
    res: dict[str,Union[int, list[int]]] = {
        "n": n,
        "lengths": lens,
        "indices": inds,
        "max_length": length+1 if y > 0 else length,
        "min_length": length,
        "num_max_length": x,
        "num_min_length": y,
        "number": m,
        }
    if isinstance(result, str):
        return res[result]
    return [ res[k] for k in result ]


T = TypeVar("T")
def partition_evenly(lst: Sequence[T], k:Optional[int]=None, m:Optional[int]=None) -> List[Sequence[T]]:
    """
    Distribute the elements of `lst` into `m` partitions, or into partitions of maximal size `k`.
     
    Actual sizes are made as large as possible, with sizes of each partition differing by at most 1.
    
    Parameters
    ----------
    lst
        sequence to partition.
    k
        maximal size of subpartitions, either k or m have to be provided, but not both.
    m
        number of subpartitions, either k or m have to be provided, but not both.
    Returns
    -------
    a list of  `m` sublists (if `m` is provided) of `lst`, with at most `k` elements (if `k` is provided) in each sublist.
    """
    indices, m = partition_lengths(len(lst), k=k, m=m, result=["indices", "number"])      # type: ignore
    return [lst[indices[i]:indices[i+1]] for i in range(m)]                               # type: ignore



def chunked(iterable: Sequence, n=None, k=None):
    """Split iterable into ``n`` iterables of similar size, or into iterables of max size ``k``.
    
    One of ``n`` and ``k`` has to be specified.

    Examples::
        >>> l = [1, 2, 3, 4]
        >>> list(chunked(l, n=4))
        [[1], [2], [3], [4]]

        >>> l = [1, 2, 3]
        >>> list(chunked(l, n=4))
        [[1], [2], [3], []]

        >>> l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> list(chunked(l, n=4))
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

    """
    if n is None:
        assert k is not None
        n = int(math.ceil(len(iterable) / k))
    else:
        assert n is not None
        k = int(math.ceil(len(iterable) / n))
    return (iterable[i * k : i * k + k] for i in range(n))


def intersperse(iterable, delimiter):
    """Return a new list where `delimiter` is inserted between every two elements in `iterable`."""

    def generator():
        it = iter(iterable)
        yield next(it)              # pylint: disable=stop-iteration-return
        for x in it:
            yield delimiter
            yield x

    return list(generator())


def expand_header(
    selector: str,
    available: Sequence[str],
    allow_missing: Union[str, bool] = False,
    single: Union[str, bool] = "warn",
) -> Union[None, str, Sequence[str]]:
    """
    Match `selector` against all elements in `available`, and return the first match (if `single`) or the list of matches.
    
    Matching uses the `fnmatch.fnmatch()` function.
    
    Parameters
    ----------
    selector
        Pattern to match in the provided list of `available` names. Wildcards "*", "?", "[chars]", "[!chars]" can be used,
        according to `fnmatch.fnmatch()` function.
    available
        list of available completion names against which `selector` is matched.
    allow_missing
        boolean or the string "warn"; if True or "warn", a `selector` failing matches with all possible values in `available`
        is allowed (but a `warnings.warn()` is shown if `allow_missing` is `warn`). Otherwise a `ValueError` is raised.
    single
        boolean or the string "warn"; if True or "warn", a `ValueError` is raised or a `warnings.warn()` shown, respectively, if
        more than one match is found. The only one/first match is returned in these cases. If False, all matches are returned.
    
    Returns
    -------
    The single/first match of the `selector` pattern (if `single`), or the list of all matches. `None` is returned if no match
    is found.
    """
    import fnmatch
    if isinstance(available, pd.Series):
        available = available.tolist()
    if len(available) > 0 and isinstance(available[0], (tuple, list)):
        matches = []
        for candidate in available:
            if (all([ fnmatch.fnmatch(cand, selector[i]) for i, cand in enumerate(candidate) ])
                if isinstance(selector, (tuple, list))
                else any([ fnmatch.fnmatch(cand, selector) for cand in candidate ])):
                matches.append(candidate)
    else:
        matches = fnmatch.filter(available, selector)
    if len(matches) < 1:
        if allow_missing == "warn":
            warnings.warn("expand_headers warning: '{0}' not found in {1}\n".format(selector, available))
        elif not allow_missing:
            raise ValueError("expand_headers: '{0}' not found in {1}".format(selector, available))
        return None
    if single:
        if len(matches) > 1:
            if single == "warn":
                warnings.warn("expand_headers warning: '{0}' found multiple times in {1} (taking first of these): {2}\n".format(
                    selector, available, matches))
            elif not allow_missing:
                raise ValueError("expand_headers: '{0}' found multiple times in {1}: {2}".format(selector, available, matches))
        return matches[0]
    return matches


def expand_headers(
    selectors: Sequence[str],
    available: Sequence[str],
    allow_missing: Union[str, bool] = False,
) -> Sequence[str]:
    """
    Match each element of `selectors` against all elements in `available`, and return the lsit of all matches found.
    
    Matching uses the `fnmatch.fnmatch()` function.
    
    Parameters
    ----------
    selectors
        List of pattern to match in the provided list of `available` names. Wildcards "*", "?", "[chars]", "[!chars]" can be
        used, according to `fnmatch.fnmatch()` function.
    available
        list of available completion names against which `selectors` elements are matched.
    allow_missing
        boolean or the string "warn"; if True or "warn", a `selector` element failing matches with all possible values in
        `available` is allowed (but a `warnings.warn()` is shown if `allow_missing` is `warn`). Otherwise a `ValueError` is
        raised if any `selectors` element does not find a match.
    
    Returns
    -------
    The single/first match of the `selector` pattern (if `single`), or the list of all matches. `None` is returned if no match
    is found.
    """
    import fnmatch
    result = []
    missing = []
    if isinstance(selectors, str):
        selectors = [selectors]
    if isinstance(available, pd.Series):
        available = available.tolist()
    for sel in selectors:
        if len(available) > 0 and isinstance(available[0], (tuple, list)):
            cur_matches = []
            for candidate in available:
                if (all([ fnmatch.fnmatch(cand, sel[i]) for i, cand in enumerate(candidate) ])
                    if isinstance(sel, (tuple, list))
                    else any([ fnmatch.fnmatch(cand, sel) for cand in candidate ])):
                    cur_matches.append(candidate)
        else:
            cur_matches = fnmatch.filter(available, sel)
        if len(cur_matches) < 1:
            missing.append(sel)
        result.extend(cur_matches)
    if len(missing) > 0:
        if allow_missing == "warn":
            warnings.warn("expand_headers: '{0}' not found in {1}".format(missing, available))
        elif not allow_missing:
            raise ValueError("expand_headers: '{0}' not found in {1}".format(missing, available))
    return result


def get_mapper(
    headers,
    single: Union[str, bool] = False,
    allow_missing: Union[str, bool] = 'warn',
):
    """
    Return a function which, when called with a str or sequence of str, returns all matches found in `headers`.
     
    See `expand_header()` and `expand_headers()` for details.
    """
    if single:
        return lambda cols: expand_header(cols, headers, allow_missing=allow_missing)
    return lambda *col: expand_headers(col[0] if (len(col) == 1) else col, headers, allow_missing=allow_missing)


def get_index(
    selector,
    available,
    allow_missing: Union[str, bool] = False,
    single: Union[str, bool] = "warn",
) -> Union[None, int, List[int]]:
    """
    Match `selector` against all elements in `available`, and return the index of the first match(es).
     
    If `single`, the only/first match index is returned, otherwise the list of
    indices of matches. Matching uses the `fnmatch.fnmatch()` function.

    Parameters
    ----------
    selector
        Pattern to match in the provided list of `available` names. Wildcards "*", "?", "[chars]", "[!chars]" can be used,
        according to `fnmatch.fnmatch()` function.
    available
        list of available completion names against which `selector` is matched.
    allow_missing
        boolean or the string "warn"; if True or "warn", a `selector` failing matches with all possible values in `available`
        is allowed (but a `warnings.warn()` is shown if `allow_missing` is `warn`). Otherwise a `ValueError` is raised.
    single
        boolean or the string "warn"; if True or "warn", a `ValueError` is raised or a `warnings.warn()` shown, respectively, if
        more than one match is found. The index of the only one/first match is returned in these cases. If False, all match
        indices are returned.

    Returns
    -------
    The index into `available` of the single/first match of the `selector` pattern (if `single`), or the list of all match
    indices. `None` is returned if no match is found.
    """
    expanded_selector = expand_header(selector, available, allow_missing, single)
    if expanded_selector is None:
        return None
    if single:
        if not expanded_selector:
            return None
        return available.index( expanded_selector )
    return [available.index(f) for f in expanded_selector]


def get_indices(selectors, available, allow_missing:Union[str,bool]=False):
    """
    Match each element of `selectors` against all elements in `available`, and return the list of indices of all matches found.
    
    Matching uses the `fnmatch.fnmatch()` function.

    Parameters
    ----------
    selectors
        List of pattern to match in the provided list of `available` names. Wildcards "*", "?", "[chars]", "[!chars]" can be
        used, according to `fnmatch.fnmatch()` function.
    available
        list of available completion names against which `selectors` elements are matched.
    allow_missing
        boolean or the string "warn"; if True or "warn", a `selector` element failing matches with all possible values in
        `available` is allowed (but a `warnings.warn()` is shown if `allow_missing` is `warn`). Otherwise a `ValueError` is
        raised if any `selectors` element does not find a match.

    Returns
    -------
    The index of the single/first match of the `selector` pattern (if `single`), or the list of indices of all matches. `None`
    is returned if no match is found.
    """
    return [available.index(f) for f in expand_headers(selectors, available, allow_missing)]


def get_indexer(
    headers,
    single: Union[str, bool] = False,
    allow_missing: Union[str, bool] = 'warn',
):
    """
    Return a function which, when called with a str or sequence of str, returns the indices of all matches found in `headers`.
    
    See `get_index()` and `get_indices()` for details.
    """
    if single:
        return lambda cols: get_index(cols, headers, allow_missing=allow_missing)
    return lambda *col: get_indices(col[0] if (len(col) == 1) else col, headers, allow_missing=allow_missing)


def map_element(func: Union[Callable,Any], lst: Sequence, index: int):
    """
    Return `lst`, where the element at position `index` is replaced by applying function `func` to it.
    
    If func is not callable, the element is replaced by `func` itself.
    """
    if hasattr(func, '__call__'):
        return [ func(el) if i == index else el for i, el in enumerate(lst) ]
    return [ func if i == index else el for i, el in enumerate(lst) ]


def unique_justseen(iterable, key=None):
    """
    List unique elements, preserving order, only remembering the element just seen.
    
    >>> unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    >>> unique_justseen('ABBCcAD', str.lower) --> A B C A D
    """
    return map(next, map(itemgetter(1), groupby(iterable, key)))


def unique_everseen(iterable, key=None):
    """
    List unique elements, preserving order, and remembering all elements ever seen.
    
    >>> unique_everseen('AAAABBBCCDAABBB') --> A B C D
    >>> unique_everseen('ABBCcAD', str.lower) --> A B C D
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def list_to_dict(lst, keys):
    """convert two lists (of elements and keys) to a dictionary."""
    return {k: val for k, val in zip(keys, lst) }


def addtodict(d, d2={}, **kwargs):      # pylint: disable=invalid-name
    """
    Add the elements of the provided 'd2' dict and/or the provided keyword arguments to a copy of the dict 'd'.
    
    A modified copy is returned.
    """
    d = copy.copy(d)
    d.update(d2, **kwargs)
    return d


def dict_remove(d, k, ignore_missing=False):
    """Remove item with key `k` from dictionary `d`. Raises `KeyError` if `k` not in `d` and `ignore_missing` is False."""
    d2 = dict(d)                        # pylint: disable=invalid-name
    if not ignore_missing or k in d2:
        del d2[k]
    return d2


def adddefaulttodict(d, d2={}, **kwargs):   # pylint: disable=invalid-name
    """Add elements of `d2` to a copy of `d`, if they are not already present."""
    d = copy.copy(d)
    for k, v in d2.items():
        if k not in d:
            d[k] = v
    for k, v in kwargs.items():
        if k not in d:
            d[k] = v
    return d


def filterdict(d, keys):
    """Return a copy of d, with all elements removed whose key is not in keys."""
    return {k: v for k, v in d.items() if k in keys}


def orderdict(d, keys):
    """
    Return an OrderedDict initialized by `d`, but with all elements removed whose key is not in `keys`.
    
    Elements will be in the order specified in keys.
    """
    res = OrderedDict()
    for key in keys:
        if key in d:
            res[key] = d[key]
    return res


def find_key(d, val):
    """
    Given a value `val`, return the key in dictionary `d` which is mapped to `val`.
    
    If no such key exists, return None.
    """
    for k, v in d.iteritems():
        if v == val:
            return k
    return None


# def dict_intersect(*dicts):
#     """"""
#     return dict( set.intersection( *( set(d.items()) for d in dicts ) ) )


class DictNamespace(dict):
    """
    Extend dicts by allowing access to their values by attribute notation.
    
    >>> d = DictNamespace()
    >>> d["a"] = 1
    >>> d.a
    -> 1
    >>> d.b = 2
    >>> d["b"]
    -> 2
    """
    # pylint: disable=too-many-instance-attributes
    def __getattr__(self, name):
        """Return dict element `name` (for names not already defined on dict)."""
        if name in self:
            return self[name]
        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        """Set dict element `name` from `value` (for names not already defined on dict)."""
        self[name] = value

    def __delattr__(self, name):
        """Remove dict element `name` (for names not already defined on dict)."""
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def break_into_cols(lst, cols):
    rows, r = divmod(len(lst), cols)
    rows += (r != 0)
    m = []
    for i in range(rows):
        m.append(lst[i::rows])
    return m

def formatcols(lst, cols, template="{:{}}", sep="    "):
    if not isinstance(lst, list):
        lst = list(lst)
    if len(lst) < 1:
        return []
    broken_table = break_into_cols(lst, cols)
    broken_table = [ flatten(line, levels=1) for line in broken_table ]
    col_lens = np.zeros([1,len(broken_table[0])]) if len(broken_table) > 0 else np.zeros([0])
    for row in broken_table:
        cur_col_lens = np.array([[ len(str(c)) for c in row ]])
        cur_col_lens.resize(col_lens.shape)
        col_lens = np.append(col_lens, cur_col_lens, axis=0)
        col_lens = np.max(col_lens, axis=0, keepdims=True)
    col_lens = col_lens[0,:].astype(int)
    template_elements = template.count("{") // 2
    return [
        sep.join([ template ] * (len(row) // template_elements )).format(*flatten(zip(row, col_lens), levels=1))
        for row in broken_table
    ]

def list_to_cols(lst, cols=3, template="{:{}}", sep="    ", p=True):
    res = "\n".join(formatcols(lst, cols=cols, template=template, sep= sep))
    if p:
        print(res)
    return res
