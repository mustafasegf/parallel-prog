#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
from collections import defaultdict


def parse_line(line, gpu=False):
    parts = line.split()
    type_name = parts[0]
    size = parts[1]
    if gpu:
        grid = parts[3]
        block = parts[5]
        time = int(parts[-1])
        config_key = f"{type_name} grid: {grid} block: {block}"
    else:
        time = int(parts[-1])
        config_key = f"{type_name} grid: 0 block: 0"  # Default grid and block for CPU
    return config_key, size, time


def read_data(filename, gpu=False):
    data = defaultdict(dict)
    sizes = set()
    with open(filename, 'r') as file:
        for line in file:
            config_key, size, time = parse_line(line, gpu)
            sizes.add(size)
            data[config_key][size] = time
    return data, sizes


def sort_sizes(sizes):
    return sorted(sizes, key=lambda x: int(x.split('x')[0]))


def merge_data(cpu_data, cpu_sizes, gpu_data, gpu_sizes):
    all_data = defaultdict(dict)
    all_sizes = cpu_sizes.union(gpu_sizes)
    # Merge CPU data
    for config, sizes_dict in cpu_data.items():
        for size, time in sizes_dict.items():
            all_data[config][size] = time
    # Merge GPU data
    for config, sizes_dict in gpu_data.items():
        for size, time in sizes_dict.items():
            if size in all_data[config]:
                all_data[config][size] = min(
                    all_data[config][size], time
                )  # Choose min time if overlapping data
            else:
                all_data[config][size] = time
    return all_data, all_sizes


def write_csv(data, sizes, filename='table.csv'):
    sorted_sizes = sort_sizes(sizes)
    fieldnames = ['configuration'] + sorted_sizes
    if filename in os.listdir():
        os.rename(filename, filename[:-4] + '_old.csv')
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for config_key, times in data.items():
            row = {'configuration': config_key}
            for size in sorted_sizes:
                row[size] = times.get(size, '')  # Fill with empty if no data
            writer.writerow(row)


def main():
    cpu_data, cpu_sizes = read_data('timecpu.txt')
    gpu_data, gpu_sizes = read_data('timegpu.txt', gpu=True)
    all_data, all_sizes = merge_data(cpu_data, cpu_sizes, gpu_data, gpu_sizes)
    write_csv(all_data, all_sizes)


if __name__ == '__main__':
    main()
