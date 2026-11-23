"""Evaluation metrics: SHD, TERE, TENE (and ECE to follow).

Purpose
-------
Structural scoring of a predicted adjacency matrix against the ground-truth
DAG (PLAN.md sections 2.5 / 7.2):
  - SHD  (Structural Hamming Distance) = missing + extra + reversed edges.
    Primary metric; direct comparison with the base paper. Lower is better.
  - TERE (True Edge Recall Error)      = true edges missed.
  - TENE (Total Edge Not Expected)     = false edges invented.
ECE (Expected Calibration Error, target < 0.1) is DivVote's new confidence
metric and will be added here later (Feb 2027, section 9).

Maps to: Week 2, Days 1-2 (PLAN.md section 10) — "Implement metrics (SHD,
TERE, TENE) with unit tests."

Implements the skeleton given in PLAN.md section 8.4.
"""

import numpy as np


def shd(pred, true):
    missing  = ((true == 1) & (pred == 0)).sum()
    extra    = ((true == 0) & (pred == 1)).sum()
    reversed_= ((true == 1) & (pred.T == 1) & (pred == 0)).sum()
    return int(missing + extra + reversed_)


def tere(pred, true):
    return int(((true == 1) & (pred == 0)).sum())


def tene(pred, true):
    return int(((true == 0) & (pred == 1)).sum())
