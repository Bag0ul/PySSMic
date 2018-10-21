from backend.manager import Manager
from backend.optimizer import Optimizer

import pandas as pd
from backend.job import Job, JobStatus
from backend.producer import Producer


def test_optimize():
    man = Manager()
    producer = Producer("id", man)
    job0 = (None, Job("id", 0, 1, pd.Series(index=[0, 5], data=[0.0, 40.0])), JobStatus.active)
    job1 = (None, Job("id", 0, 2, pd.Series(index=[0, 2], data=[0.0, 20.0])), JobStatus.active)
    job2 = (None, Job("id", 0, 3, pd.Series(index=[2, 5], data=[0.0, 30.0])), JobStatus.active)
    producer.schedule = [job0, job1, job2]
    producer.prediction = pd.Series(index=[0, 5], data=[0.0, 50.0])

    result = producer.optimizer.optimize()

    assert result[0] == 0
    assert result[1] == 1
    assert result[2] == 1


def test_differentiate():
    o = Optimizer(None)
    series1 = pd.Series(index=[0, 3600], data=[0.0, 1.0])
    expected1 = pd.Series(index=[0, 3600], data=[0, 1.0])
    actual1 = o.differentiate(series1)

    series2 = pd.Series(index=[0, 1, 2, 3, 4, 5], data=[0.0, 0.5, 1.0, 2.0, 2.0, 3.0])
    expected2 = pd.Series(index=[0, 1, 2, 3, 4, 5], data=[0.0, 0.5, 0.5, 1.0, 0.0, 1.0])
    actual2 = o.differentiate(series2)

    assert all(list(expected1 == actual1))
    assert all(list(expected2 == actual2))


def test_interpolate():
    o = Optimizer(None)
    series = pd.Series(index=[2, 4, 6], data=[4.0, 8.0, 12.0])
    actual = o.interpolate(series, [0, 1, 2, 3, 4, 5, 6, 7, 8]).round(4)
    expected = pd.Series(index=[0, 1, 2, 3, 4, 5, 6, 7, 8], data=[4.0, 4.0, 4.0, 6.0, 8.0, 10.0, 12.0, 12.0, 12.0])
    assert all(list(actual.data == expected))
