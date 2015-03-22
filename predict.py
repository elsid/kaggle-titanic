#!/usr/bin/env python3
# coding: utf-8

import re
import yaml

from argparse import ArgumentParser, FileType
from collections import Counter
from functools import reduce
from itertools import permutations
from numpy import unique
from operator import itemgetter
from pandas import read_csv
from sklearn.ensemble import AdaBoostClassifier
from sys import stdin, stdout

NAME_PATTERN = re.compile(r'^[^,]+, (?P<title>\w+)')

COLUMNS = (
    'Age',
    'Cabin',
    'Embarked',
    'Fare',
    'Name',
    'Parch',
    'PassengerId',
    'Pclass',
    'Sex',
    'SibSp',
    'Ticket',
    'Title',
)

BASE_PERCENTAGES_COLUMNS = (
    'Pclass',
    'Sex',
    'SibSp',
    'Parch',
    'Title',
    'Embarked',
)

PERCENTAGES_COLUMNS = (
    'Pclass',
    'Sex',
    ('Sex', 'Pclass'),
    ('Sex', 'SibSp', 'Parch'),
    ('Sex', 'Title'),
)


def print_table(dictionary):
    for k, v in sorted(dictionary.items(), key=itemgetter(1)):
        print('{:>30} {:>20}'.format(str(k), v))


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


def prepare_train_data(data, percentages_columns):
    add_title(data)
    fill_age(data)
    fill_embarked(data)
    fill_pclass(data)
    fill_fare(data)
    add_percentages(data, data, percentages_columns)


def prepare_test_data(data, train_data, percentages_columns):
    add_title(data)
    fill_age(data, train_data)
    fill_embarked(data, train_data)
    fill_pclass(data)
    fill_fare(data, train_data)
    add_percentages(data, train_data, percentages_columns)
    if 'Survived' in data.columns:
        data.drop('Survived', axis=1, inplace=True)


def drop_unused(data, used_columns):
    return data.drop(frozenset(COLUMNS) - frozenset(used_columns), axis=1)


def classify(test_values, train_values):
    classifier = AdaBoostClassifier(n_estimators=200)
    classifier = classifier.fit(train_values[0:, 1:], train_values[0:, 0])
    return classifier.predict(test_values).astype(int)


def predict(test_data, train_data, percentages_columns, used_columns):
    prepare_train_data(train_data, percentages_columns)
    prepare_test_data(test_data, train_data, percentages_columns)
    test_data['Survived'] = classify(
        drop_unused(test_data, used_columns).values,
        drop_unused(train_data, used_columns).values)


def get_prediction_precision(test_data, sample_data):
    assert (tuple(x for x in test_data.PassengerId.values)
            == tuple(x for x in sample_data.PassengerId.values))
    return (sum(p == s for p, s in zip(test_data.Survived.values,
                                       sample_data.Survived.values))
            / len(test_data.Survived.values))


def percentages(data, max_count, percentages_columns):
    def generate_columns():
        for n in range(1, max_count + 1):
            for column_ in permutations(BASE_PERCENTAGES_COLUMNS, n):
                yield column_

    prepare_train_data(data, percentages_columns)
    for column in generate_columns():
        print(column)
        print_table(survivors_percentage(data, column))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('train', type=FileType('r'))
    parser.add_argument('test', type=FileType('r'), nargs='?', default=stdin)
    parser.add_argument('-s', '--sample', type=FileType('r'))
    parser.add_argument('-p', '--print_percentages', type=int)
    parser.add_argument('-c', '--percentages_columns', type=FileType('r'))
    parser.add_argument('-u', '--use_column', dest='used_columns', type=str,
                        nargs='+', default=tuple())
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = read_csv(args.train, header=0)
    percentages_columns = (yaml.load(args.percentages_columns)
                           if args.percentages_columns else PERCENTAGES_COLUMNS)
    if args.print_percentages:
        percentages(train_data, args.print_percentages, percentages_columns)
    else:
        test_data = read_csv(args.test, header=0)
        predict(test_data, train_data, percentages_columns, args.used_columns)
        if args.sample:
            sample_data = read_csv(args.sample, header=0)
            print(get_prediction_precision(test_data, sample_data))
        else:
            test_data.set_index('PassengerId').to_csv(stdout,
                                                      columns=('Survived',))


if __name__ == '__main__':
    main()
