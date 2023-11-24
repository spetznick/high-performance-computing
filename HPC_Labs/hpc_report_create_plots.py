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


count = counter()
# Ping pong plots


filenames = [
    "pingpong_n1.out",
    "pingpong_n2.out"
]
dir = "HPC/out/"

message_sizes = []
message_times = []

for filename in filenames:
    message_sizes_f = []
    message_times_f = []
    with open(dir + filename, newline='') as csvfile:
        linereader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(linereader):
            if i == 0 or i == 1:
                continue
            message_sizes_f.append(float(row[1]))
            message_times_f.append(1000*float(row[0]))
    message_sizes.append(message_sizes_f)
    message_times.append(message_times_f)

# print(message_sizes)
# print(message_times)

# in bytes assuming int is 32 bits
message_sizes_bytes = np.array(message_sizes)/4
message_times_numpy = np.array(message_times)/1000

A_linReg = np.vstack(
    [message_sizes_bytes[0][1:], np.ones(len(message_sizes_bytes[0][1:]))]).T

beta, alpha = np.linalg.lstsq(
    A_linReg, message_times_numpy[0][1:], rcond=None)[0]
print(f"t_(m) = alpha + beta * m = {alpha} + {beta} * m")

pingpong_fig = plt.figure(counter.count())
ax = plt.gca()
ax.set_title('time of ping-pong over message size on 1 node')
ax.set_xlabel('message size in bytes')
ax.set_ylabel('time of communication in [ms]')
ax.scatter(message_sizes_bytes[0],
           message_times[0], marker='x', s=30, alpha=0.9, label='measurements')
ax.plot(message_sizes_bytes[0], 1000*(alpha + beta *
        message_sizes_bytes[0]), 'cyan', label='linear regression')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid()
ax.legend()
pingpong_fig.savefig('figs/pingpong-doublelog_n1.svg')

pingpong_fig = plt.figure(counter.count())
ax = plt.gca()
ax.set_title('time of ping-pong over message size on two nodes')
ax.set_xlabel('message size in bytes')
ax.set_ylabel('time of communication in [ms]')
ax.scatter(message_sizes_bytes[1],
           message_times[1], marker='x', s=30, alpha=0.9)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid()
pingpong_fig.savefig('figs/pingpong-doublelog_n2.svg')

# MM-product plots

# consider using https://github.com/kirbs-/hide_code to hide specific cells or
# use second python script for creating plots

processes = [1, 2, 8, 24, 48, 64]
number_of_runs = 5
number_of_sizes = 3
filenames = ['MM-product_p01_.out',
             'MM-product_p02_.out',
             'MM-product_p08_.out',
             'MM-product_p24_.out',
             'MM-product_p48_.out',
             'MM-product_p64_.out'
             ]
dir = 'HPC/out/'

times = []
array_sizes = []
number_of_processes = []

for filename in filenames:
    times_f = []
    array_sizes_f = []
    number_of_processes_f = []

    with open(dir + filename, newline='') as csvfile:
        linereader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(linereader):
            if i == 0:
                continue
            array_sizes_f.append(float(row[0]))
            number_of_processes_f.append(float(row[1]))
            times_f.append(float(row[2]))

    times.append(times_f)
    array_sizes.append(array_sizes_f)
    number_of_processes.append(number_of_processes_f)

# print(array_sizes)
# print(number_of_processes)
# print(times)

# plots:
# box plot:
# plot times
figure = plt.figure(counter.count())
ax = plt.gca()
ax.scatter(array_sizes[0][0:15], times[0][0:15], marker='x', s=30, alpha=0.3)
ax.scatter(array_sizes[1][0:15], times[1][0:15], marker='x', s=30, alpha=0.3)
ax.scatter(array_sizes[2][0:15], times[2][0:15], marker='x', s=30, alpha=0.3)
ax.set_ylabel('time of computation in [s]')
ax.set_title('number of processors: 1')
ax.grid()
ax.set_xlabel('array size')

# time over number processes, matrix size 1000
figure_time_over_nprocess_1000 = plt.figure(counter.count())
ax = plt.gca()
ax.set_title('matrix size 1000x1000')
ax.set_xlabel('number of processes')
ax.set_ylabel('time of computation in [s]')
ax.scatter(number_of_processes[0][11:15], times[0]
           [11:15], marker='x', s=30, alpha=0.3)
ax.scatter(number_of_processes[1][11:15], times[1]
           [11:15], marker='x', s=30, alpha=0.3)
ax.scatter(number_of_processes[2][11:15], times[2]
           [11:15], marker='x', s=30, alpha=0.3)
ax.scatter(number_of_processes[3][11:15], times[3]
           [11:15], marker='x', s=30, alpha=0.3)
ax.scatter(number_of_processes[4][11:15], times[4]
           [11:15], marker='x', s=30, alpha=0.3)
ax.scatter(number_of_processes[5][11:15], times[5]
           [11:15], marker='x', s=30, alpha=0.3)
ax.grid()
figure_time_over_nprocess_1000.savefig(
    'figs/MM-product-m1000-time-over-procs.svg')
figure_time_over_nprocess_1000.savefig(
    'figs/MM-product-m1000-time-over-procs.pdf')
figure_time_over_nprocess_1000.savefig(
    'figs/MM-product-m1000-time-over-procs.png')

# time over number processes, all matrix sizes
figure_time_over_nprocess = plt.figure(counter.count())
ax = plt.gca()
ax.set_xlabel('number of processes')
ax.set_ylabel('time of computation in [s]')
for p in range(len(processes)):
    ax.scatter(number_of_processes[p][0:5], times[p][0:5], marker='x',
               s=30, alpha=0.3, color='r', label='Matrix size 10')
    ax.scatter(number_of_processes[p][5:10], times[p][5:10],
               marker='x', s=30, alpha=0.3, color='g', label='Matrix size 100')
    ax.scatter(number_of_processes[p][11:15], times[p][11:15], marker='x',
               s=30, alpha=0.3, color='cyan', label='Matrix size 1000')
    if p == 0:
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                  loc="lower left",
                  mode="expand",
                  fancybox=True, shadow=True, ncol=len(processes))

ax.set_xticks(processes)
# ax.set_xscale('log')
ax.set_yscale('log')
figure_time_over_nprocess.savefig('figs/MM-product-time-over-procs.svg')
figure_time_over_nprocess.savefig('figs/MM-product-time-over-procs.pdf')
figure_time_over_nprocess.savefig('figs/MM-product-time-over-procs.png')
plt.show()
