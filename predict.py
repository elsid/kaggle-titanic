#!/usr/bin/env python3
# coding: utf-8

import re
import yaml

from argparse import ArgumentParser, FileType
from collections import Counter
from functools import reduce
from numpy import unique
from pandas import read_csv
from sklearn.ensemble import AdaBoostClassifier
from sys import stdin, stdout

NAME_PATTERN = re.compile(r'^[^,]+, (?P<title>\w+)')


def fill_age(data, train_data=None):
    train_data = data if train_data is None else train_data
    median_age = train_data.Age.dropna().median()
    data.loc[data.Age.isnull(), 'Age'] = median_age


def fill_embarked(data, train_data=None):
    train_data = data if train_data is None else train_data
    data.loc[data.Embarked.isnull(), 'Embarked'] = (
        train_data.Embarked.dropna().mode().values)


def fill_pclass(data):
    data.loc[data.Pclass.isnull(), 'Pclass'] = 'unknown'


def fill_fare(data, train_data=None):
    train_data = data if train_data is None else train_data
    median_fare = {
        pclass: train_data[train_data.Pclass == pclass].Fare.dropna()
                                                            .median() or 0
        for pclass in unique(train_data.Pclass)}
    for pclass, val in median_fare.items():
        data.loc[data.Fare.isnull() & (data.Pclass == pclass), 'Fare'] = val


def count_at(train_data, columns):
    return Counter(tuple(x) for x in train_data[list(columns)].values)


def count_sum_at(train_data, columns):
    values = train_data[list(columns)].values

    def count(counter, value):
        counter[tuple(value[:-1])] += value[-1]
        return counter

    return reduce(count, values, Counter())


def count_survivors(train_data, column):
    if isinstance(column, str):
        return count_sum_at(train_data, (column, 'Survived'))
    else:
        return count_sum_at(train_data, list(column) + ['Survived'])


def survivors_percentage(train_data, column):
    survivors_count = count_survivors(train_data, column)
    if isinstance(column, str):
        all_count = count_at(train_data, (column,))
    else:
        all_count = count_at(train_data, column)
    return {k: survivors_count[k] / v for k, v in all_count.items()}


def add_percentages(data, train_data, percentages_columns):
    train_data = data if train_data is None else train_data
    for column in percentages_columns:
        if isinstance(column, str):
            column = (column,)
        percentage = survivors_percentage(train_data, column)

        def from_percentage(x):
            return percentage[x] if x in percentage else 0.5

        def get_percents():
            values = (tuple(x) for x in data[list(column)].values)
            return [from_percentage(x) for x in values]

        percents = get_percents()
        data[tuple(list(column) + ['Percentage'])] = percents


def parse_title(name):
    match = NAME_PATTERN.search(name)
    return match.group('title')


def add_title(data):
    data['Title'] = [parse_title(name) for name in data.Name.values]


def fill_train_data(data):
    add_title(data)
    fill_age(data)
    fill_embarked(data)
    fill_pclass(data)
    fill_fare(data)


def prepare_train_data(data, percentages_columns):
    fill_train_data(data)
    add_percentages(data, data, percentages_columns)


def prepare_test_data(data, train_data, percentages_columns):
    if 'Survived' in data.columns:
        data.drop('Survived', axis=1, inplace=True)
    add_title(data)
    columns = data.columns
    fill_age(data, train_data)
    fill_embarked(data, train_data)
    fill_pclass(data)
    fill_fare(data, train_data)
    add_percentages(data, train_data, percentages_columns)
    return columns


def drop_unused(data, unused_columns):
    return data.drop(frozenset(unused_columns), axis=1)


def classify(test_values, train_values):
    classifier = AdaBoostClassifier(n_estimators=200)
    classifier = classifier.fit(train_values[0:, 1:], train_values[0:, 0])
    return classifier.predict(test_values).astype(int)


def predict(test_data, train_data, percentages_columns, used_columns):
    prepare_train_data(train_data, percentages_columns)
    columns = prepare_test_data(test_data, train_data, percentages_columns)
    unused_columns = columns - used_columns
    test_data['Survived'] = classify(
        drop_unused(test_data, unused_columns).values,
        drop_unused(train_data, unused_columns).values)


def get_prediction_precision(test_data, sample_data):
    assert (tuple(x for x in test_data.PassengerId.values)
            == tuple(x for x in sample_data.PassengerId.values))
    return (sum(p == s for p, s in zip(test_data.Survived.values,
                                       sample_data.Survived.values))
            / len(test_data.Survived.values))


class Config(object):
    def __init__(self, data):
        self.used_columns = (data['used_columns'] if 'used_columns' in data
                             else [])
        self.percentages_columns = (data['percentages_columns']
                                    if 'percentages_columns' in data else [])


def parse_config(stream):
    data = yaml.load(stream)
    return Config(data)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', type=FileType('r'))
    parser.add_argument('train', type=FileType('r'))
    parser.add_argument('test', type=FileType('r'), nargs='?', default=stdin)
    parser.add_argument('-s', '--sample', type=FileType('r'))
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = read_csv(args.train, header=0)
    config = parse_config(args.config)
    test_data = read_csv(args.test, header=0)
    predict(test_data, train_data, config.percentages_columns,
            config.used_columns)
    if args.sample:
        sample_data = read_csv(args.sample, header=0)
        print(get_prediction_precision(test_data, sample_data))
    else:
        test_data.set_index('PassengerId').to_csv(stdout,
                                                  columns=('Survived',))


if __name__ == '__main__':
    main()
