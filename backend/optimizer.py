import itertools
import logging
import math
from collections import defaultdict
from itertools import chain
from random import random, shuffle, randint
from typing import List

import numpy as np
import pandas as pd
from scipy import optimize


class Optimizer:
    def __init__(self, producer):
        self.producer = producer
        self.logger = logging.getLogger("src.Optimizer")
        self.differentiated_loads = []
        self.differentiated_production = None
        self.indices = []
        self.current_config = None
        self.penalty_factor = 1.0
        self.min_value = float('inf')

    # The main function that optimizes the schedule. How the schedule and job should be implemented is up for discussion
    def optimize(self):
        indices = list(set(chain.from_iterable(
            map(lambda x: self.producer.schedule[x][1].load_profile.index.values.tolist(),
                range(0, len(self.producer.schedule))))))
        indices.sort()
        self.indices = indices

        # Interpolate and differentiate load profiles before optimization.
        self.differentiated_production = self.differentiate_and_interpolate(self.producer.prediction, indices)
        for s in self.producer.schedule:
            self.differentiated_loads.append(self.differentiate_and_interpolate(s[1].load_profile, indices))

        objective = np.zeros(len(self.producer.schedule))
        configs = list(map(list, itertools.product([0, 1], repeat=len(self.differentiated_loads))))
        shuffle(configs)
        return self.recursive_binary_optimizer(configs, float('inf'), configs[0])

    def to_minimize(self, schedule: List[float]):
        if len(list(filter(lambda x: math.isnan(x), schedule))) > 0:
            return float('inf')

        produced = defaultdict(float)
        consumed = defaultdict(float)
        current_loads = [l for i, l in enumerate(self.differentiated_loads) if self.current_config[i] > 0]

        for t in self.indices:
            for i, load in enumerate(current_loads):
                consumed[t + int(schedule[i])] += load[t]
            produced[t] += self.differentiated_production[t]

        ts = list(set(produced.keys()).union(consumed.keys()))
        sorted(ts)

        produced_values = []
        consumed_values = []
        for t in ts:
            if t in produced:
                produced_values.append(produced[t])
            else:
                produced_values.append(np.nan)
            if t in consumed:
                consumed_values.append(consumed[t])
            else:
                consumed_values.append(np.nan)
        produced_series = pd.Series(index=ts, data=produced_values).interpolate(method="barycentric")
        consumed_series = pd.Series(index=ts, data=consumed_values).interpolate(method="barycentric")

        penalty = 0
        diff = 0
        for p, c in zip(list(produced_series.values), list(consumed_series.values)):
            diff_t = abs(p - c)
            diff += diff_t
            if c > p:
                penalty += diff_t

        if penalty > self.min_value:
            return float('inf')

        return diff + penalty

    def binary_optimzier(self):
        configs = map(list, itertools.product([0, 1], repeat=len(self.differentiated_loads)))
        min_config = None
        min_config_value = float('inf')
        for c in configs:
            diff, penalty = self.to_minimize(c)
            if diff < min_config_value:
                min_config_value = diff + penalty
                min_config = c
        return min_config

    def recursive_binary_optimizer(self, configs, min_value, min_config):
        if len(configs) == 0:
            return min_config
        self.current_config = configs[0]
        bounds = [(s[1].est, s[1].lst) for i, s in enumerate(self.producer.schedule) if self.current_config[i] >= 1]
        s = [randint(s[1].est, s[1].lst) for i, s in enumerate(self.producer.schedule) if self.current_config[i] >= 1]
        if len(s) > 0:
            result = optimize.minimize(self.to_minimize, s, tol=0.0, method='L-BFGS-B', bounds=bounds)
        else:
            return self.recursive_binary_optimizer(configs[1:], min_value, min_config)

        diff = result.fun
        if diff == float('inf'):
            config_indices = [i for i, x in enumerate(self.current_config) if x == 1.0]
            new_configs = list(filter(lambda c: not all(map(lambda i: c[i], config_indices)), configs))
            return self.recursive_binary_optimizer(new_configs, min_value, min_config)

        elif diff < min_value:
            self.min_value = diff
            return self.recursive_binary_optimizer(configs[1:], diff, self.current_config)

        else:
            return self.recursive_binary_optimizer(configs[1:], min_value, min_config)

    def differentiate_and_interpolate(self, series: pd.Series, indices: List[int]):
        interpolated = self.interpolate(series, indices)
        derivative = self.differentiate(interpolated)
        return derivative

    def differentiate(self, series: pd.Series):
        indices = list(series.index)
        values = list(series.values)
        derivative = []
        for i in range(1, len(indices) + 1):
            if i >= len(indices):
                derivative.insert(0, values[0])
            else:
                p0 = values[i - 1]
                t0 = indices[i - 1]
                p1 = values[i]
                t1 = indices[i]

                d = (p1 - p0)
                derivative.append(d)

        return pd.Series(index=indices, data=derivative)

    def interpolate(self, series: pd.Series, indices: List[int]):
        not_included = list(filter(lambda t: t not in series.index.values, indices))
        with_new_indices = series.append(pd.Series(index=not_included, data=[np.nan for x in range(len(not_included))]))
        interpolated = with_new_indices.interpolate(method="barycentric").sort_index()

        min_index = min(series.index.values)
        max_index = max(series.index.values)
        start_value = series.data[0]
        end_value = series.data[-1]

        for t, p in interpolated.items():
            if t < min_index:
                interpolated.update(pd.Series(data=[start_value], index=[t]))
            elif t > max_index:
                interpolated.update(pd.Series(data=[end_value], index=[t]))

        return interpolated
