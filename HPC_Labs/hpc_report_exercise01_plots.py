import matplotlib.pyplot as plt
import numpy as np
import csv
plt.close('all')


class counter:
    def __init__(self) -> None:
        counter.val = 0

    def count():
        c = counter.val
        counter.val += 1
        return c


"""
- 4 1 1.93 100x100
Number of iterations: 131, omega: 1.93
Delta: 0.000066
Delta: 0.000089
Delta: 0.000099
Delta: 0.000078
(1) Elapsed Wtime       0.005309 s ( 89.1% CPU)
(2) Elapsed Wtime       0.005308 s ( 85.5% CPU)
(3) Elapsed Wtime       0.005310 s ( 86.7% CPU)
(0) Elapsed Wtime       0.009293 s ( 89.2% CPU)

- 2 2 1.93 100x100
Number of iterations: 131, omega: 1.93
Delta: 0.000089
Delta: 0.000082
Delta: 0.000099
Delta: 0.000087
(1) Elapsed Wtime       0.005498 s ( 89.0% CPU)
(2) Elapsed Wtime       0.005496 s ( 89.0% CPU)
(3) Elapsed Wtime       0.005501 s ( 90.1% CPU)
(0) Elapsed Wtime       0.009459 s ( 90.4% CPU)

- 4 1 1.93 200x200
Number of iterations: 532, omega: 1.93
Delta: 0.000063
Delta: 0.000098
Delta: 0.000099
Delta: 0.000078
(1) Elapsed Wtime       0.059708 s ( 94.3% CPU)
(2) Elapsed Wtime       0.059707 s ( 94.5% CPU)
(3) Elapsed Wtime       0.059712 s ( 94.2% CPU)
(0) Elapsed Wtime       0.074123 s ( 94.9% CPU)

- 2 2 1.93 200x200
Number of iterations: 532, omega: 1.93
Delta: 0.000098
Delta: 0.000096
Delta: 0.000099
Delta: 0.000098
(1) Elapsed Wtime       0.060286 s ( 98.1% CPU)
(2) Elapsed Wtime       0.060288 s ( 97.5% CPU)
(3) Elapsed Wtime       0.060292 s ( 97.8% CPU)
(0) Elapsed Wtime       0.075101 s ( 97.1% CPU)

- 4 1 1.93 400x400
Number of iterations: 1561, omega: 1.93
Delta: 0.000066
Delta: 0.000099
Delta: 0.000100
Delta: 0.000077
(1) Elapsed Wtime       0.619254 s ( 98.8% CPU)
(2) Elapsed Wtime       0.619255 s ( 99.0% CPU)
(3) Elapsed Wtime       0.619252 s ( 98.9% CPU)
(0) Elapsed Wtime       0.673155 s ( 99.0% CPU)

- 2 2 1.93 400x400
Number of iterations: 1561, omega: 1.93
Delta: 0.000099
Delta: 0.000098
Delta: 0.000100
Delta: 0.000099
(1) Elapsed Wtime       0.620277 s ( 99.3% CPU)
(2) Elapsed Wtime       0.620275 s ( 99.1% CPU)
(3) Elapsed Wtime       0.620277 s ( 99.0% CPU)
(0) Elapsed Wtime       0.674249 s ( 99.3% CPU)

- 4 1 1.93 800x800
Number of iterations: 3601, omega: 1.93
Delta: 0.000071
Delta: 0.000100
Delta: 0.000100
Delta: 0.000072
(1) Elapsed Wtime       5.472918 s ( 99.7% CPU)
(2) Elapsed Wtime       5.472918 s ( 99.7% CPU)
(3) Elapsed Wtime       5.472930 s ( 99.7% CPU)
(0) Elapsed Wtime       5.691494 s ( 99.7% CPU)

- 2 2 1.93 800x800
Number of iterations: 3601, omega: 1.93
Delta: 0.000100
Delta: 0.000100
Delta: 0.000100
Delta: 0.000100
(1) Elapsed Wtime       5.486153 s ( 99.7% CPU)
(2) Elapsed Wtime       5.486152 s ( 99.8% CPU)
(3) Elapsed Wtime       5.486152 s ( 99.8% CPU)
(0) Elapsed Wtime       5.705497 s ( 99.7% CPU)
"""
