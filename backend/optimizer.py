import itertools
import logging
import math
from collections import defaultdict
from enum import Enum
from itertools import chain
from random import randint, random
from typing import List

import numpy as np
import pandas as pd
from scipy import optimize
import util.optimizer_utils as utils


class Algorithm(Enum):
    """The currently possible algorithms."""
    fifty_fifty = 'fifty_fifty'
    L_BFGS_B = 'L-BFGS-B'
    SLSQP = 'SLSQP'
    TNC = 'TNC'


class Optimizer:
    def __init__(self, producer, options):
        self.producer = producer
        self.options = options
        self.logger = logging.getLogger("src.Optimizer")
        self.differentiated_loads = []
        self.min_objective_value = float('inf')
        self.penalty_factor = options["penalty_factor"] if "penalty_factor" in self.options else 0.5
        self.algorithm = Algorithm[options["algo"]]

        # The estimated total produced energy is the last entry of the producer's prediction profile
        self.objection_value_offset = 0

    def optimize(self):
        if self.algorithm in [Algorithm.L_BFGS_B, Algorithm.SLSQP, Algorithm.TNC]:
            schedule_times, should_keep = self.basinhopping()

        elif self.algorithm == Algorithm.fifty_fifty:
            schedule_times, should_keep = self.fifty_fifty()

        else:
            should_keep = [True] * len(self.producer.schedule)
            schedule_times = [s['job'].scheduled_time for s in self.producer.schedule]

        return schedule_times, should_keep

    def basinhopping(self):
        self.reset_and_differentiate_loads()
        self.objection_value_offset = self.producer.prediction.values[-1]
        tol = self.options["tol"] if "tol" in self.options else 1000.0
        eps = self.options["eps"] if "eps" in self.options else 0.001
        now = self.producer.manager.clock.now
        bounds = [(max(s['job'].est, now), s['job'].lst) for s in self.producer.schedule]
        x0 = np.array([randint(b[0], b[1]) for b in bounds])
        kwargs = dict(method=self.algorithm.value, bounds=bounds, options=dict(eps=eps), tol=tol)

        # The Basin-hopping algorithm is good at finding global minimums.
        result = optimize.basinhopping(func=self.objective_function, x0=x0, minimizer_kwargs=kwargs)
        objective_value = result.fun
        schedule_times = list([int(round(e)) for e in result.x])

        should_keep = [True for x in range(len(self.producer.schedule))]

        if np.isinf(self.min_objective_value):
            # If the solution we have found is bad, and no previous solution has been discovered.
            if not self.strictly_positive():
                should_keep[-1] = False
            else:
                self.min_objective_value = objective_value
        elif objective_value < self.min_objective_value:
            self.min_objective_value = objective_value
        else:
            should_keep[-1] = False

        return schedule_times, should_keep

    def fifty_fifty(self):
        should_keep = [True for x in range(len(self.producer.schedule))]
        need_grid_power = not self.strictly_positive()
        if random() < 0.5 or need_grid_power:
            should_keep[-1] = False
        schedule_times = [s['job'].scheduled_time for s in self.producer.schedule]
        return schedule_times, should_keep

    def objective_function(self, schedule: List[float]):
        """The function we want to minimize. schedule is a List of start times for the jobs in the schedule."""
        consumed = defaultdict(float)
        for i, load in enumerate(self.differentiated_loads):
            for t, p in load.items():
                if not math.isnan(schedule[i]):
                    start_time = schedule[i]
                    # Add power, p(t), to consumed at time 'start_time' plus 't'
                    consumed[int(round(start_time + t))] += p

        # Differentiate and interpolate production profile on the indices found in consumer
        ts = list(consumed.keys())
        produced = utils.differentiate_and_interpolate(self.producer.prediction, ts)
        ts.extend(list(produced.index.values))
        ts = sorted(list(set(ts)))

        total_consumed = 0
        penalty = 0
        for i, t in enumerate(ts):
            if i == len(ts) - 1:
                delta = 0
            else:
                delta = abs(ts[i+1] - ts[i])

            p = produced[t]
            c = consumed[t]

            # Multiply difference between produced and consumed in time 't' by the distance to the previous time index.
            # This way, differences are weighted based on their time span.
            total_consumed += c * delta

            # If consumed is greater than produced, we have negative amount of production power.
            if c > p:
                diff_t = abs(p - c) * delta
                penalty += diff_t

        # Return the difference between produced and consumed energy, in addition to the penalty for having negative
        # power levels. Finally, subtract this from the objection_value_offset, i.e. the sum of the produced power. This
        # is so that the theoretically minimal value is equal to zero (if we consume all produced energy).
        score = abs(self.objection_value_offset - total_consumed) + penalty * self.penalty_factor
        return score

    def strictly_positive(self):
        """Checks if the current schedules yields positive power in every time step (i.e. we always have more produced
        energy than consumed energy. Mostly the same as the objective function, but terminates faster, and returns a
        boolean value indicating whether the excess energy function is always positive, given the current schedule."""

        schedule = [s['job'].scheduled_time for s in self.producer.schedule]
        self.reset_and_differentiate_loads()
        consumed = defaultdict(float)
        for i, load in enumerate(self.differentiated_loads):
            for t, p in load.items():
                offset = schedule[i]
                consumed[int(round(t + offset))] += p

        ts = list(consumed.keys())
        produced = utils.differentiate_and_interpolate(self.producer.prediction, ts)
        ts.extend(list(produced.index.values))
        ts = sorted(ts)
        for i, t in enumerate(ts):
            if consumed[t] > produced[t]:
                return False
        return True

    #
    def reset_and_differentiate_loads(self):
        """Re-compute the differentiated and interpolated load profiles"""

        indices = list(set(chain.from_iterable(
            map(lambda x: self.producer.schedule[x]['job'].load_profile.index.values.tolist(),
                range(0, len(self.producer.schedule))))))
        indices.sort()

        # Interpolate and differentiate load profiles before optimization.
        self.differentiated_loads = []
        for s in self.producer.schedule:
            self.differentiated_loads.append(utils.differentiate(s['job'].load_profile))
