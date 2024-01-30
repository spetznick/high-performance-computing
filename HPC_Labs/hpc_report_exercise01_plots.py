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

# 1.2.3


def exercise_1_2_3():
    time_per_iterations = [
        [(1, 4), [0.000043,
                  0.000042,
                  0.000042,
                  0.000042,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000043,
                  0.000042,
                  0.000042,
                  0.000042,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000043,
                  0.000042,
                  0.000042,
                  0.000042,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000043,
                  0.000042,
                  0.000042,
                  0.000042,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041,
                  0.000041], 100],
        [(2, 2), [0.000044,
                  0.000043,
                  0.000042,
                  0.000044,
                  0.000042,
                  0.000043,
                  0.000042,
                  0.000043,
                  0.000042,
                  0.000042,
                  0.000044,
                  0.000043,
                  0.000042,
                  0.000044,
                  0.000042,
                  0.000043,
                  0.000042,
                  0.000043,
                  0.000042,
                  0.000042,
                  0.000044,
                  0.000043,
                  0.000042,
                  0.000044,
                  0.000042,
                  0.000043,
                  0.000042,
                  0.000043,
                  0.000042,
                  0.000042,
                  0.000044,
                  0.000043,
                  0.000042,
                  0.000044,
                  0.000042,
                  0.000043,
                  0.000042,
                  0.000043,
                  0.000042,
                  0.000042], 100],
        [(4, 1), [0.000039,
                  0.000038,
                  0.000037,
                  0.000038,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000039,
                  0.000038,
                  0.000037,
                  0.000038,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000039,
                  0.000038,
                  0.000037,
                  0.000038,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000039,
                  0.000038,
                  0.000037,
                  0.000038,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037,
                  0.000037], 100],
        [(1, 4), [0.000157,
                  0.000149,
                  0.000148,
                  0.000148,
                  0.000157,
                  0.000149,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000157,
                  0.000149,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000157,
                  0.000149,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148,
                  0.000148], 200],
        [(2, 2), [0.000152,
                  0.000152,
                  0.000151,
                  0.000155,
                  0.000155,
                  0.000152,
                  0.000152,
                  0.000152,
                  0.000152,
                  0.000150,
                  0.000152,
                  0.000152,
                  0.000151,
                  0.000155,
                  0.000155,
                  0.000152,
                  0.000152,
                  0.000152,
                  0.000152,
                  0.000150,
                  0.000152,
                  0.000152,
                  0.000151,
                  0.000155,
                  0.000155,
                  0.000152,
                  0.000152,
                  0.000152,
                  0.000152,
                  0.000150,
                  0.000152,
                  0.000152,
                  0.000151,
                  0.000155,
                  0.000155,
                  0.000152,
                  0.000152,
                  0.000152,
                  0.000152,
                  0.000150], 200],
        [(4, 1), [0.000185,
                  0.000156,
                  0.000156,
                  0.000156,
                  0.000155,
                  0.000156,
                  0.000156,
                  0.000155,
                  0.000157,
                  0.000156,
                  0.000185,
                  0.000156,
                  0.000156,
                  0.000156,
                  0.000155,
                  0.000156,
                  0.000182,
                  0.000156,
                  0.000156,
                  0.000156,
                  0.000155,
                  0.000156,
                  0.000156,
                  0.000155,
                  0.000157,
                  0.000156,
                  0.000156,
                  0.000155,
                  0.000157,
                  0.000156,
                  0.000182,
                  0.000156,
                  0.000157,
                  0.000156,
                  0.000155,
                  0.000156,
                  0.000156,
                  0.000155,
                  0.000157,
                  0.000156], 200],
        [(1, 4), [0.000546,
                  0.000561,
                  0.000558,
                  0.000560,
                  0.000565,
                  0.000560,
                  0.000561,
                  0.000560,
                  0.000565,
                  0.000559,
                  0.000546,
                  0.000561,
                  0.000558,
                  0.000560,
                  0.000565,
                  0.000560,
                  0.000561,
                  0.000560,
                  0.000565,
                  0.000559,
                  0.000546,
                  0.000561,
                  0.000558,
                  0.000560,
                  0.000565,
                  0.000560,
                  0.000561,
                  0.000560,
                  0.000565,
                  0.000559,
                  0.000546,
                  0.000561,
                  0.000558,
                  0.000560,
                  0.000565,
                  0.000560,
                  0.000561,
                  0.000560,
                  0.000565,
                  0.000559], 400],
        [(2, 2), [0.000602,
                  0.000604,
                  0.000591,
                  0.000600,
                  0.000598,
                  0.000600,
                  0.000597,
                  0.000596,
                  0.000597,
                  0.000600,
                  0.000602,
                  0.000603,
                  0.000591,
                  0.000600,
                  0.000598,
                  0.000600,
                  0.000597,
                  0.000596,
                  0.000597,
                  0.000600,
                  0.000602,
                  0.000603,
                  0.000591,
                  0.000600,
                  0.000598,
                  0.000600,
                  0.000597,
                  0.000596,
                  0.000597,
                  0.000600,
                  0.000602,
                  0.000603,
                  0.000591,
                  0.000600,
                  0.000598,
                  0.000600,
                  0.000597,
                  0.000596,
                  0.000597,
                  0.000600], 400],
        [(4, 1), [0.000552,
                  0.000556,
                  0.000553,
                  0.000555,
                  0.000563,
                  0.000604,
                  0.000589,
                  0.000583,
                  0.000575,
                  0.000581,
                  0.000552,
                  0.000556,
                  0.000553,
                  0.000555,
                  0.000563,
                  0.000604,
                  0.000589,
                  0.000583,
                  0.000575,
                  0.000581,
                  0.000552,
                  0.000556,
                  0.000553,
                  0.000555,
                  0.000563,
                  0.000604,
                  0.000589,
                  0.000583,
                  0.000575,
                  0.000581,
                  0.000552,
                  0.000556,
                  0.000553,
                  0.000555,
                  0.000563,
                  0.000604,
                  0.000589,
                  0.000583,
                  0.000575,
                  0.000581], 400],
        [(1, 4), [0.002214,
                  0.002217,
                  0.002230,
                  0.002202,
                  0.002261,
                  0.002183,
                  0.002212,
                  0.002260,
                  0.002203,
                  0.002272,
                  0.002213,
                  0.002217,
                  0.002230,
                  0.002201,
                  0.002261,
                  0.002183,
                  0.002212,
                  0.002260,
                  0.002203,
                  0.002272,
                  0.002213,
                  0.002217,
                  0.002230,
                  0.002201,
                  0.002261,
                  0.002183,
                  0.002212,
                  0.002260,
                  0.002203,
                  0.002272,
                  0.002213,
                  0.002217,
                  0.002230,
                  0.002201,
                  0.002261,
                  0.002183,
                  0.002212,
                  0.002260,
                  0.002203,
                  0.002272], 800],
        [(2, 2), [0.002102,
                  0.002099,
                  0.002102,
                  0.002096,
                  0.002104,
                  0.002099,
                  0.002102,
                  0.002096,
                  0.002106,
                  0.002103,
                  0.002101,
                  0.002100,
                  0.002103,
                  0.002102,
                  0.002102,
                  0.002099,
                  0.002102,
                  0.002096,
                  0.002106,
                  0.002103,
                  0.002101,
                  0.002100,
                  0.002103,
                  0.002102,
                  0.002106,
                  0.002103,
                  0.002101,
                  0.002100,
                  0.002103,
                  0.002102,
                  0.002104,
                  0.002099,
                  0.002102,
                  0.002096,
                  0.002106,
                  0.002103,
                  0.002101,
                  0.002100,
                  0.002103,
                  0.002102], 800],
        [(4, 1), [0.002325,
                  0.002320,
                  0.002318,
                  0.002325,
                  0.002325,
                  0.002320,
                  0.002318,
                  0.002325,
                  0.002312,
                  0.002317,
                  0.002312,
                  0.002317,
                  0.002317,
                  0.002325,
                  0.002317,
                  0.002325,
                  0.002305,
                  0.002325,
                  0.002305,
                  0.002325,
                  0.002325,
                  0.002320,
                  0.002318,
                  0.002325,
                  0.002312,
                  0.002317,
                  0.002317,
                  0.002325,
                  0.002305,
                  0.002325,
                  0.002325,
                  0.002320,
                  0.002318,
                  0.002325,
                  0.002312,
                  0.002317,
                  0.002317,
                  0.002325,
                  0.002305,
                  0.002325], 800]
    ]

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))

    # grid size 100
    all_data = []
    x_axis = [""]
    for (a, b), liste, grid_size in time_per_iterations:
        if grid_size == 100:
            print(f"({a},{b}): {grid_size}")
            x_axis.append(f"({a},{b}): {grid_size}")
            all_data.append([1000*el for el in liste])

    axs[0, 0].boxplot(all_data)
    axs[0, 0].xaxis.set_ticks([i for i in range(len(x_axis))], labels=x_axis)
    axs[0, 0].semilogy()
    axs[0, 0].set_ylabel("time per iteration [ms]")
    axs[0, 0].grid()

    # grid size 200
    all_data = []
    x_axis = [""]
    for (a, b), liste, grid_size in time_per_iterations:
        if grid_size == 200:
            print(f"({a},{b}): {grid_size}")
            x_axis.append(f"({a},{b}): {grid_size}")
            all_data.append([1000*el for el in liste])

    axs[0, 1].boxplot(all_data)
    axs[0, 1].xaxis.set_ticks([i for i in range(len(x_axis))], labels=x_axis)
    axs[0, 1].semilogy()
    axs[0, 1].grid()

    # grid size 400
    all_data = []
    x_axis = [""]
    for (a, b), liste, grid_size in time_per_iterations:
        if grid_size == 400:
            print(f"({a},{b}): {grid_size}")
            x_axis.append(f"({a},{b}): {grid_size}")
            all_data.append([1000*el for el in liste])

    axs[1, 0].semilogy()
    axs[1, 0].boxplot(all_data)
    axs[1, 0].xaxis.set_ticks([i for i in range(len(x_axis))], labels=x_axis)
    axs[1, 0].set_ylabel("time per iteration [ms]")
    axs[1, 0].grid()

    # grid size 800
    all_data = []
    x_axis = [""]
    for (a, b), liste, grid_size in time_per_iterations:
        if grid_size == 800:
            print(f"({a},{b}): {grid_size}")
            x_axis.append(f"({a},{b}): {grid_size}")
            all_data.append([1000*el for el in liste])

    axs[1, 1].semilogy()
    axs[1, 1].boxplot(all_data)
    axs[1, 1].xaxis.set_ticks([i for i in range(len(x_axis))], labels=x_axis)
    axs[1, 1].grid()

    plt.show()
    # fig.savefig("HPC_Labs/figs/exercise01_ex_1_2_3.pdf")
    # fig.savefig("HPC_Labs/figs/exercise01_ex_1_2_3.svg")

    all_data = np.array(all_data).flatten()
    average_800_grid = np.average(all_data)
    print(f"average on 800x800: {average_800_grid}")

    # 1.2.5

    N = [100**2, 200**2, 400**2, 800**2]
    vals = [131, 532, 1561, 3601]

    f = plt.figure()
    ax = plt.gca()
    ax.scatter(N, vals)
    ax.set_title("number of iterations over grid size")
    ax.set_ylabel("# iterations")
    ax.set_xlabel("grid size")
    # f.savefig("HPC_Labs/figs/exercise01_ex_1_2_5.pdf")
    # f.savefig("HPC_Labs/figs/exercise01_ex_1_2_5.svg")

# 1.2.6


def exercise_1_2_6():
    # read file:
    errors = np.zeros(5000)
    with open("HPC_Labs/HPC/exercise01/errors_0.dat") as file:
        for i, line in enumerate(file):
            if i == 0:
                title = line.rstrip()
            else:
                errors[i-1] = float(line.rstrip())

    f = plt.figure()
    ax = plt.gca()
    ax.plot(errors, label=title)
    ax.set_title("error over # iterations")
    ax.set_xlabel("# iterations")
    ax.set_ylabel("global error")
    plt.semilogy()
    plt.grid()
    plt.legend()
    plt.show()
    f.savefig("HPC_Labs/figs/exercise01_ex_1_2_6.pdf")
    f.savefig("HPC_Labs/figs/exercise01_ex_1_2_6.svg")

# 1.2.9


def exercise_1_2_9():
    total_times_normal = [5.469, 5.463, 5.447, 5.446, 5.461,
                          5.468, 5.498, 5.451, 5.455, 5.447, 5.439, 5.448, 5.446]
    total_times_no_parity = [4.336, 4.421, 4.391,
                             4.348, 4.504, 4.315, 4.399, 4.388, 4.293, 4.276,
                             4.288, 4.291, 4.287, 4.334, 4.312, 4.304, 4.320, 4.345, 4.401]
    joint_data = [total_times_normal, total_times_no_parity]

    fig, axs = plt.subplots(nrows=1, ncols=2)

    x_axis = [f"with parity check (# runs: {len(total_times_normal)})",
              f"no parity check (# runs: {len(total_times_no_parity)})"]

    axs[0].boxplot(total_times_normal, labels=[x_axis[0]])
    axs[0].set_ylabel("total time [s]")

    axs[1].boxplot(total_times_no_parity, labels=[x_axis[1]])
    avg_time = sum(total_times_normal)/len(total_times_normal)
    avg_time_no_parity = sum(total_times_no_parity)/len(total_times_no_parity)

    print("Avg times: ", avg_time, avg_time_no_parity,
          int(100*(1 - avg_time_no_parity/avg_time)), '%')

    plt.show()

    fig.savefig("HPC_Labs/figs/exercise01_ex_1_2_9.pdf")
    fig.savefig("HPC_Labs/figs/exercise01_ex_1_2_9.svg")


def exercise_1_2_13():
    fig = plt.figure(figsize=(16, 8))
    ax = plt.gca()
    data_100 = [0.00005,
                0.00006,
                0.00005,
                0.00005,
                0.00006,
                0.00005,
                0.00006,
                0.00006,
                0.00006,
                0.00006]
    data_200 = [0.00021,
                0.00021,
                0.00021,
                0.00021,
                0.00021,
                0.00021,
                0.00022,
                0.00021,
                0.00021,
                0.00021]
    data_400 = [0.00083,
                0.00081,
                0.00080,
                0.00081,
                0.00081,
                0.00081,
                0.00080,
                0.00080,
                0.00081,
                0.00082]
    data_800 = [0.00324,
                0.00320,
                0.00320,
                0.00319,
                0.00319,
                0.00321,
                0.00322,
                0.00322,
                0.00327,
                0.00322]
    data_1600 = [
        0.01272,
        0.01273,
        0.01273,
        0.01272,
        0.01275,
        0.01274,
        0.01271,
        0.01272,
        0.01273,
        0.01273
    ]
    # grid size 100
    all_data = [data_100, data_200, data_400, data_800, data_1600]
    x_axis = ["", '100', '200', '400', '800', '1600']

    ax.scatter(100*np.ones(len(all_data[0])), all_data[0])
    ax.scatter(200*np.ones(len(all_data[1])), all_data[1])
    ax.scatter(400*np.ones(len(all_data[2])), all_data[2])
    ax.scatter(800*np.ones(len(all_data[3])), all_data[3])
    ax.scatter(1600*np.ones(len(all_data[4])), all_data[4])
    # ax.xaxis.set_ticks([i for i in range(len(x_axis))], labels=x_axis)
    ax.semilogy()
    ax.set_ylabel("time per iteration [ms]")
    ax.set_xlabel("size n")
    # ax.grid()

    plt.show()
    fig.savefig("HPC_Labs/figs/exercise01_ex_1_2_13.pdf")
    fig.savefig("HPC_Labs/figs/exercise01_ex_1_2_13.svg")


if __name__ == "__main__":
    exercise_1_2_13()
