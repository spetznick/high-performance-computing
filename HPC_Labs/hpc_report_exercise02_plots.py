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


def oneTo2d(i):
    j = i % 3
    k = i//3
    return j, k


def results_reader(filename):
    data = dict()
    with open(file=filename, mode='r') as f:
        for i, line in enumerate(f):
            if line == '\n':
                continue
            if 'Number of' in line:
                continue
            if i == 0:
                labels = [el.strip() for el in line.rsplit(',')][4:]
            else:
                line_el = line.rsplit(',')
                px = line_el[1]
                py = line_el[2]
                n = line_el[3]
                key = (px, py, n)
                if key in data:
                    times = [[float(el)] for el in line_el[5:]]
                    old_list = data[key]
                    data[key] = list(map(list.__add__, old_list, times))
                else:
                    times = [[float(el)] for el in line_el[5:]]
                    data[key] = times

    return labels, data


def exercise_4_2():
    labels, all_data = results_reader(
        'HPC_Labs/HPC/exercise02/results_ex_4_2_sc.txt')
    labels = labels[1:]
    # fig = plt.figure(figsize=(16, 8))
    # ax = plt.gca()

    # grid size 100

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    i = 0
    for (px, py, n), times in all_data.items():

        # Example data
        y_pos = np.arange(len(labels))

        # sum_time = sum(times[1:])

        # percentages = [val/sum_time for val in times[1:]]
        xtick = np.ones(len(labels))

        ax[i//3, i % 3].scatter(0*xtick, times[0])
        ax[i//3, i % 3].scatter(1*xtick, times[1])
        ax[i//3, i % 3].scatter(2*xtick, times[2])
        ax[i//3, i % 3].scatter(3*xtick, times[3])
        ax[i//3, i % 3].set_xticks(y_pos)
        ax[i//3, i % 3].set_xticklabels([])
        if i % 3 == 0:
            ax[i//3, i % 3].set_ylabel('time [s]')
        if i // 3 == 1:
            ax[i//3, i % 3].set_xticklabels(labels=labels, rotation=30)
        ax[i//3, i % 3].set_title(f'(px, py, n)=({px},{py},{n})')
        i += 1

    plt.show()

    fig.savefig("HPC_Labs/figs/exercise02_ex_4_2.pdf")
    fig.savefig("HPC_Labs/figs/exercise02_ex_4_2.svg")


def exercise_4_5():
    labels, all_data = results_reader(
        'HPC_Labs/HPC/exercise02/results_ex_4_5.txt')
    labels = labels[1:]
    # fig = plt.figure(figsize=(16, 8))
    # ax = plt.gca()

    # grid size 100

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    i = 0
    for (px, py, n), times in all_data.items():

        # Example data
        y_pos = np.arange(len(labels))

        # sum_time = sum(times[1:])

        # percentages = [val/sum_time for val in times[1:]]
        xtick = np.ones(len(labels))

        ax.scatter(0*xtick, times[0])
        ax.scatter(1*xtick, times[1])
        ax.scatter(2*xtick, times[2])
        ax.scatter(3*xtick, times[3])
        ax.set_xticks(y_pos)
        ax.set_xticklabels([])
        ax.set_ylabel('time [s]')
        ax.set_xticklabels(labels=labels, rotation=30)
        ax.set_title(f'(px, py, n)=({px},{py},{n})')
        i += 1

    fig.savefig("HPC_Labs/figs/exercise02_ex_4_5.pdf")
    fig.savefig("HPC_Labs/figs/exercise02_ex_4_5.svg")


def exercise_4_6():
    labels, all_data = results_reader(
        'HPC_Labs/HPC/exercise02/results_ex_4_6.txt')
    labels = labels[1:]

    fig, ax = plt.subplots(1, 3, figsize=(16, 8))
    i = 0
    for (px, py, n), times in all_data.items():

        # Example data
        y_pos = np.arange(len(labels))

        # sum_time = sum(times[1:])

        # percentages = [val/sum_time for val in times[1:]]
        xtick = np.ones(len(labels))

        ax[i % 3].scatter(0*xtick, times[0])
        ax[i % 3].scatter(1*xtick, times[1])
        ax[i % 3].scatter(2*xtick, times[2])
        ax[i % 3].scatter(3*xtick, times[3])
        ax[i % 3].set_xticks(y_pos)
        ax[i % 3].set_xticklabels([])
        if i % 3 == 0:
            ax[i % 3].set_ylabel('time [s]')
        ax[i % 3].set_xticklabels(labels=labels, rotation=30)
        ax[i % 3].set_title(f'(px, py, n)=({px},{py},{n})')
        i += 1

    plt.show()

    fig.savefig("HPC_Labs/figs/exercise02_ex_4_6.pdf")
    fig.savefig("HPC_Labs/figs/exercise02_ex_4_6.svg")


if __name__ == "__main__":
    # exercise_4_2()
    # exercise_4_5()
    exercise_4_6()
