import os
import math
import numpy as np

from fpdf import FPDF
import matplotlib.pyplot as plt


def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    x = x - mean
    x = x / std
    return x


def dataset_report(report_dir, x, y, groups):
    os.makedirs(report_dir, exist_ok=True)

    img_dir = os.path.join(report_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    pdf = FPDF()

    img_files = plot_all_groups(x, y, groups, img_dir)
    for img_file in img_files:
        pdf.add_page()
        pdf.image(img_file, w=100)

    pdf_file = os.path.join(report_dir, 'dataset_report.pdf')
    pdf.output(pdf_file)


def plot_all_groups(x, y, groups, img_dir):
    group_ids = np.unique(groups)
    img_files = []
    for gid in group_ids:
        img_file = plot_group(x, y, groups, gid, img_dir)
        img_files.append(img_file)

    return img_files


def plot_group(x, y, groups, group_id, img_dir):
    idx_group = groups == group_id
    xg = x[idx_group, :, :, :]

    n_samples = xg.shape[0]
    n_imgs = 16
    n_rows = int(n_imgs / 2)
    n_cols = int(4)
    step = math.floor(n_samples / n_imgs)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, constrained_layout=True)
    fig.set_size_inches(w=10, h=20)

    idx_plot = 1
    idx_sample = 0
    for r in range(int(n_rows / 2)):
        for c in range(n_cols):
            print(f'prep plot {r*2, c}')
            ax = axs[r*2, c]
            img = xg[idx_sample, :, :, 0]
            img = normalize(img)
            pos = ax.imshow(img, origin='lower', aspect='auto')
            ax.set_title(f'#{idx_sample}, {y[idx_sample]}')
            fig.colorbar(pos, ax=ax)

            print(f'prep plot {r*2+1, c}')
            ax = axs[r*2+1, c]
            img = xg[idx_sample, :, :, 1]
            img = normalize(img)
            pos = ax.imshow(img, origin='lower', aspect='auto')
            fig.colorbar(pos, ax=ax)

            idx_sample += step
            idx_plot += 1

        idx_plot += 4

    img_file = os.path.join(img_dir, f'{group_id}.png')
    plt.savefig(img_file, dpi=300)
    return img_file
