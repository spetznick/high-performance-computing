import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_data(file_path):
    data = np.loadtxt(file_path, delimiter=' ', skiprows=0)
    x = data[:, 0]
    y = data[:, 1]
    val = data[:, 2]
    return x, y, val


def read_data(file_path):
    data = np.loadtxt(file_path, delimiter=' ', skiprows=0)
    # x = data[:, 0]
    # y = data[:, 1]
    # val = data[:, 2]
    val_map = np.zeros(
        (int(np.max(data[:, 0])), int(np.max(data[:, 1]))))
    for x, y, val in data:
        val_map[int(y)-1, int(x)-1] = val
    return val_map


def plot_scatter(x, y, val, subplot):
    sc = subplot.scatter(x, y, c=val, cmap='viridis',
                         edgecolors='k', linewidths=0.5)
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
    plt.colorbar(sc, ax=subplot)
    subplot.set_title('Scatter Plot for Laplace Equation')


def plot_contour(val_map, subplot):
    sc = subplot.imshow(val_map, cmap='viridis')
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
    subplot.set_title('Scatter Plot for Sequential Poisson Solver')
    plt.colorbar(sc, ax=subplot)


def plot_combined_scatter(file_paths, subplot):
    data_stack = None
    for i, file_path in enumerate(file_paths):
        data = np.loadtxt(file_path, delimiter=' ', skiprows=0)
        if i == 0:
            data_stack = data
        else:
            data_stack = np.vstack([data_stack, data])
    val_map = np.zeros(
        (int(np.max(data_stack[:, 0])), int(np.max(data_stack[:, 1]))))
    for x, y, val in data_stack:
        val_map[int(y)-1, int(x)-1] = val

    # x = data_stack[:, 0]
    # y = data_stack[:, 1]
    # val = data_stack[:, 2]
    # sc = subplot.scatter(x, y, c=val, cmap='viridis',
    #                      edgecolors='k', linewidths=0.5)
    # data_stack[:, 2] = np.linalg.norm(data_stack[:, 2], axis=0, ord=2)
    sc = subplot.imshow(val_map, cmap='viridis')
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
    subplot.set_title('Combined Plot for Parallel Poisson Solver')
    plt.colorbar(sc, ax=subplot)


def main():
    parser = argparse.ArgumentParser(
        description='Plot scatter plot for Laplace equation')
    parser.add_argument('file_path', type=str,
                        help='Path to the txt-file with data')
    parser.add_argument('file_pattern', nargs='+', type=str,
                        help='File pattern with wildcard, e.g., "output*.dat"')
    args = parser.parse_args()
    fig, subplots = plt.subplots(1, 2)
    try:
        val_map = read_data(args.file_path)
        plot_contour(val_map, subplots[0])
    except Exception as e:
        print(f"Error: {e}")

    file_paths = args.file_pattern
    if not file_paths:
        print(f"No files found for pattern: {args.file_pattern}")
        return

    plot_combined_scatter(file_paths, subplots[1])
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()


if __name__ == "__main__":
    main()
