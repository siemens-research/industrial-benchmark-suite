"""
Preallocated channel history of a CCBench episode.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT

Author:
    Holger Schoener <holger.schoener@siemens.com>
"""

from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Mapping

import numpy as np
import pandas as pd


class EpisodeHistory(Mapping[str, pd.Series]):
    """Float histories of the channels of one episode on a regular time index, preallocated for speed.

    Appending to a `pd.Series` copies it, which made long episodes slow. Here, values are written into preallocated
    arrays and the mapping returns Series views of the rows written so far. The length is tracked per channel, as the
    channels of a timestep are computed one after the other: a generator reading a channel that has not yet been
    computed for the current timestep sees the values up to the previous timestep, as with growing Series. The
    capacity is doubled when exceeded.

    Args:
        start: Timestamp of the first row.
        freq: Duration of one timestep.
        capacity: Number of rows to preallocate.
        channels: Names of the channels to create.
    """

    def __init__(self, start: pd.Timestamp, freq: pd.Timedelta, capacity: int, channels: Iterable[str] = ()):
        self._start = start
        self._freq = freq
        self._index = self._make_index(max(int(capacity), 1))
        self._values: Dict[str, np.ndarray] = {}
        self._lengths: Dict[str, int] = {}
        for name in channels:
            self.add_channel(name)

    def _make_index(self, capacity: int) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._start + self._freq * np.arange(capacity))

    @property
    def capacity(self) -> int:
        """Number of preallocated rows per channel."""
        return len(self._index)

    def add_channel(self, name: str) -> None:
        """Create an empty channel history.

        Args:
            name: Name of the channel.
        """
        self._values[name] = np.full(self.capacity, np.nan)
        self._lengths[name] = 0

    def append(self, name: str, timestamp: pd.Timestamp, value: float) -> None:
        """Append the value of a channel for the next timestep.

        Args:
            name: Name of the channel.
            timestamp: Timestamp of the value, must be the next timestamp of the regular index for this channel.
            value: Value to append.
        """
        row = self._lengths[name]
        if row >= self.capacity:
            self._grow()
        if self._index[row] != timestamp:
            raise ValueError(f"expected timestamp {self._index[row]} for row {row} of {name}, got {timestamp}")
        self._values[name][row] = value
        self._lengths[name] = row + 1

    def _grow(self) -> None:
        capacity = 2 * self.capacity
        self._index = self._make_index(capacity)
        for name, values in self._values.items():
            self._values[name] = np.concatenate([values, np.full(capacity - len(values), np.nan)])

    def __getitem__(self, name: str) -> pd.Series:
        length = self._lengths[name]
        return pd.Series(self._values[name][:length], index=self._index[:length], name=name, copy=False)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)
