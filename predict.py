#!/usr/bin/env python3
# coding: utf-8

import re
import yaml

from argparse import ArgumentParser, FileType
from collections import Counter, Iterable
from functools import reduce
from numpy import unique
from pandas import read_csv, concat
from sklearn.ensemble import AdaBoostClassifier
from sys import stdin, stdout

NAME_PATTERN = re.compile(r'^[^,]+, (?P<title>\S+)')


def fill_age(data, train_data=None):
    train_data = data if train_data is None else train_data
    median_age = train_data.Age.dropna().median()
    data.loc[data.Age.isnull(), 'Age'] = median_age


def fill_embarked(data, train_data=None):
    train_data = data if train_data is None else train_data
    data.loc[data.Embarked.isnull(), 'Embarked'] = (
        train_data.Embarked.dropna().mode().values)


def fill_fare(data, train_data=None):
    train_data = data if train_data is None else train_data
    median_fare = {
        pclass: train_data[train_data.Pclass == pclass].Fare.dropna().median()
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


def add_relatives_count(data):
    data['RelativesCount'] = data['SibSp'] + data['Parch']


def add_columns(data):
    add_title(data)
    add_relatives_count(data)


def fill_train_data(data):
    fill_age(data)
    fill_embarked(data)
    fill_fare(data)


def prepare_train_data(data, percentages_columns):
    fill_train_data(data)
    add_percentages(data, data, percentages_columns)


def prepare_test_data(data, train_data, percentages_columns):
    if 'Survived' in data.columns:
        data.drop('Survived', axis=1, inplace=True)
    columns = data.columns
    fill_age(data, train_data)
    fill_embarked(data, train_data)
    fill_fare(data, train_data)
    add_percentages(data, train_data, percentages_columns)
    return columns


def drop_unused(data, unused_columns):
    return data.drop(frozenset(unused_columns), axis=1)


def classify(test_values, train_values):
    classifier = AdaBoostClassifier(n_estimators=200)
    classifier = classifier.fit(train_values[0:, 1:], train_values[0:, 0])
    return classifier.predict(test_values).astype(int)


def predict(config):
    if config.splits:
        for split in config.splits:
            predict(split)
        config.test_data = concat([x.test_data for x in config.splits])
    else:
        prepare_train_data(config.train_data, config.percentages_columns)
        columns = prepare_test_data(config.test_data, config.train_data,
                                    config.percentages_columns)
        unused_columns = columns - config.used_columns
        config.test_data['Survived'] = classify(
            drop_unused(config.test_data, unused_columns).values,
            drop_unused(config.train_data, unused_columns).values)


def get_prediction_precision(test_data, sample_data):
    test_data = test_data.sort('PassengerId')
    sample_data = sample_data.sort('PassengerId')
    assert (tuple(test_data.PassengerId.values)
            == tuple(sample_data.PassengerId.values))
    return (sum(p == s for p, s in zip(test_data.Survived.values,
                                       sample_data.Survived.values))
            / len(test_data.Survived.values))


class Config(object):
    def __init__(self, config_data, train_data, test_data):
        self.used_columns = (
            config_data['used_columns']
            if 'used_columns' in config_data else [])
        self.percentages_columns = (
            config_data['percentages_columns']
            if 'percentages_columns' in config_data else [])
        self.train_data = train_data
        self.test_data = test_data
        self.splits = [split for split in self._generate_splits(config_data)]

    def _generate_splits(self, config_data):
        columns = frozenset(list(self.train_data.columns)
                            + ['Title', 'RelativesCount'])
        for key, value in config_data.items():
            if key in columns:
                for column_value in value:
                    if isinstance(column_value, dict):
                        yield self._on_dict(key, column_value)
                    else:
                        yield self._on_value(key, column_value)

    def _on_dict(self, column_name, column_value):
        assert len(tuple(column_value.keys())) == 1
        value = tuple(column_value.keys())[0]
        column_dict = column_value[value]
        if 'used_columns' in column_dict:
            column_dict['used_columns'] += self.used_columns
        else:
            column_dict['used_columns'] = self.used_columns
        if 'percentages_columns' in column_dict:
            column_dict['percentages_columns'] += self.percentages_columns
        else:
            column_dict['percentages_columns'] = self.percentages_columns
        return Config(column_dict,
                      self.train_data.loc[
                          self.train_data[column_name] == value].copy(),
                      self.test_data.loc[
                          self.test_data[column_name] == value].copy())

    def _on_value(self, column_name, value):
        column_dict = {
            'used_columns': self.used_columns,
            'percentages_columns': self.percentages_columns,
        }
        if isinstance(value, str) or not isinstance(value, Iterable):
            return Config(column_dict,
                          self.train_data.loc[
                              self.train_data[column_name] == value].copy(),
                          self.test_data.loc[
                              self.test_data[column_name] == value].copy())
        else:
            return Config(column_dict,
                          self.train_data.loc[
                              self.train_data[column_name].isin(
                                  frozenset(value))].copy(),
                          self.test_data.loc[
                              self.test_data[column_name].isin(
                                  frozenset(value))].copy())


def parse_config(stream, train_data, test_data):
    return Config(yaml.load(stream), train_data, test_data)


def parse_args():
    parser = ArgumentParser(
        description='Predicts survivors on Titanic ' + '=' * 78 + ' '
                    'See http://www.kaggle.com/c/titanic-gettingStarted')
    parser.add_argument('config', type=FileType('r'),
                        help='path to config file in special yaml format, '
                             'see config.yaml')
    parser.add_argument('train', type=FileType('r'),
                        help='path to file in csv format with information '
                             'about passengers contains an indication of '
                             'survival')
    parser.add_argument('test', type=FileType('r'), nargs='?', default=stdin,
                        help='path to file in csv format with information '
                             'about passengers without indication of '
                             'survival, by default uses stdin')
    parser.add_argument('-s', '--sample', type=FileType('r'),
                        help='path to file in csv format with information '
                             'about passengers survival, if set script prints '
                             'precision of prediction')
    parser.add_argument('-f', '--full', action='store_true',
                        help='output csv file with all columns')
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = read_csv(args.train, header=0)
    test_data = read_csv(args.test, header=0)
    add_columns(train_data)
    add_columns(test_data)
    config = parse_config(args.config, train_data, test_data)
    predict(config)
    if args.sample:
        sample_data = read_csv(args.sample, header=0)
        print(get_prediction_precision(config.test_data, sample_data))
    else:
        result = config.test_data.sort('PassengerId').set_index('PassengerId')
        if args.full:
            result.to_csv(stdout)
        else:
            result.to_csv(stdout, columns=('Survived',))


if __name__ == '__main__':
    main()
