#!/bin/bash
aws s3 sync run_baseline/estimator s3://galaxy-zoo/active-learning/runs/run_baseline/estimator
aws s3 sync run/estimator s3://galaxy-zoo/active-learning/runs/run/estimator