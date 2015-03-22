#!/usr/bin/env python3
# coding: utf-8


from argparse import ArgumentParser, FileType
from itertools import combinations
from operator import itemgetter
from pandas import read_csv
from sys import stdin

from predict import fill_train_data, survivors_percentage


def print_table(dictionary):
    for k, v in sorted(dictionary.items(), key=itemgetter(1)):
        print('{:>30} {:>20}'.format(str(k), v))


def percentages(data, max_count, columns):
    fill_train_data(data)

    def generate_columns():
        for n in range(1, max_count + 1):
            for column_ in combinations(columns, n):
                yield column_

    for column in generate_columns():
        print(column)
        print_table(survivors_percentage(data, column))


def parse_args():
    parser = ArgumentParser(
        description='prints percentages tables for all combinations of base '
                    'columns limited by max count')
    parser.add_argument('train', type=FileType('r'), default=stdin,
                        help='path to train data in csv file, stdin uses by '
                             'default')
    parser.add_argument('max_count', type=int,
                        help='max number of columns in one combination')
    parser.add_argument('base_column', type=str, nargs='+',
                        help='names of columns to use in combinations')
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = read_csv(args.train, header=0)
    percentages(train_data, args.max_count, args.base_column)

if __name__ == '__main__':
    main()
