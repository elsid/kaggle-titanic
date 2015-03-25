#!/usr/bin/env python3
# coding: utf-8

import pylab

from argparse import ArgumentParser, FileType
from predict import (
    read_train_data, parse_config, prepare_train_data, drop_unused)


def hist(config):
    if config.splits:
        for split in config.splits:
            hist(split)
    else:
        columns = frozenset(config.train_data.columns) - frozenset(['Survived'])
        prepare_train_data(config.train_data, config.percentages_columns)
        unused_columns = columns - frozenset(config.used_columns)
        data = drop_unused(config.train_data, unused_columns)
        for x in data.hist():
            for y in x:
                y.title.set_text('.'.join(config.path) + ' '
                                 + y.title.get_text())


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', type=FileType('r'),
                        help='path to config file in special yaml format, '
                             'see config.yaml')
    parser.add_argument('data', type=FileType('r'),
                        help='path to file in csv format with information '
                             'about passengers')
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = read_train_data(args.data)
    config = parse_config(args.config, train_data)
    hist(config)
    pylab.show()


if __name__ == '__main__':
    main()
