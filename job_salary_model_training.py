import pandas as pd
import numpy as np
df = pd.read_csv('job_salary_data/job_salary_data.csv')
df.shape
df.head()

# Missing data
df.count()

from collections import Counter
Counter(df.ContractType)
Counter(df.ContractTime)
df.SalaryNormalized.median()

feature_registry = {} #To be filled with features below

def feature_fn(f):
    """
    Decorator for feature functions. Adds function to registry, and does some basic type checking
    or arguments and return values
    """
    def decorator(raw_data):
        assert type(raw_data) == pd.DataFrame, "Bad argument type"
        res = f(raw_data)
        assert (type(res) == pd.DataFrame) or (type(res) == pd.Series), "Bad return value type"
        return res

    feature_registry[f.__name__] = decorator
    return decorator

def design_matrix(raw_data, feature_registry_names):
    assert len(set(feature_registry_names)) == len(feature_registry_names),  \
        "Duplicate feature names detected"
    assert set(feature_registry_names).issubset(set(feature_registry.keys())), \
        "Unknown feature fn name %s" % (set(feature_registry_names) - set(feature_registry.keys()))

    feature_dataframes = []
    for feature_fn_name in feature_registry_names:
        feature_dataframes.append(feature_registry[feature_fn_name](raw_data))

    return pd.concat(feature_dataframes, axis=1)

def labels(raw_data):
    return (raw_data.SalaryNormalized > 30000).astype(int)
train_test_split_index = int(12200 * 0.7) # 70% of data for training, 30% of data for testing
train_data = df.iloc[:train_test_split_index]
test_data = df.iloc[train_test_split_index:]

from collections import defaultdict
category_median_salary_dict = defaultdict(lambda: 30000)
category_median_salary_dict.update(train_data[['Category', 'SalaryNormalized']].groupby('Category').median().to_dict()['SalaryNormalized'])

@feature_fn
def category_median_salary(raw_data):
    return pd.DataFrame(data=[category_median_salary_dict[row['Category']] for _, row in raw_data.iterrows()],
                        index=raw_data.index,
                        columns=['category_median_salary'])


CONTRACT_TYPES = ['nan', 'part_time', 'full_time']
@feature_fn
def contract_type_one_hot(df):
    res = pd.DataFrame(index=df.index)
    for contract_type in CONTRACT_TYPES:
        res['contract_type_%s' % contract_type] = (df.ContractType.astype(str) == contract_type).astype(int)
    return res

CONTRACT_TIMES = ['nan', 'permanent', 'contract']
@feature_fn
def contract_time_one_hot(df):
    res = pd.DataFrame(index=df.index)
    for contract_type in CONTRACT_TIMES:
        res['contract_time_%s' % contract_type] = (df.ContractTime.astype(str) == contract_type).astype(int)
    return res
from collections import Counter
low_salary_counter = Counter(' '.join(train_data[train_data.SalaryNormalized <= 30000].Title).lower().split())
high_salary_counter = Counter(' '.join(train_data[train_data.SalaryNormalized > 30000].Title).lower().split())
low_salary_counter_frequent = Counter(dict([(k,v) for (k,v) in low_salary_counter.iteritems() if v > 10]))
high_salary_counter_frequent = Counter(dict([(k,v) for (k,v) in high_salary_counter.iteritems() if v > 10]))

title_words = set(low_salary_counter_frequent.keys()).union(set(high_salary_counter_frequent.keys()))
rel_freq = {}
for word in title_words:
    rel_freq[word] = ((high_salary_counter[word] / float(sum(train_data.SalaryNormalized > 30000))) /
                     ((low_salary_counter[word] + 0.01) / float(sum(train_data.SalaryNormalized <= 30000))))

HIGH_SALARY_WORDS = {key for key, value in rel_freq.iteritems() if value > 1}
LOW_SALARY_WORDS = {key for key, value in rel_freq.iteritems() if value <= 1}
def row_num_high_salary_words(row):
    return len([word for word in row['Title'].lower().split() if word in HIGH_SALARY_WORDS])

@feature_fn
def num_high_salary_words(df):
    return pd.DataFrame(data=df.apply(row_num_high_salary_words, axis=1),
                        columns=['num_high_salary_words'])

def row_num_low_salary_words(row):
    return len([word for word in row['Title'].lower().split() if word in LOW_SALARY_WORDS])

@feature_fn
def num_low_salary_words(df):
    return pd.DataFrame(data=df.apply(row_num_low_salary_words, axis=1),
                        columns=['num_low_salary_words'])

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


import hashlib
import datetime
import pickle

def model_description(model, model_config, raw_data):
    res = {}
    res['model_config'] = model_config
    res['data_sha'] = hashlib.sha256(str(raw_data.describe)).hexdigest()

    if type(model) == LogisticRegression:
        res['description'] = str(zip(raw_data.columns, list(model.coef_[0])))
    elif type(model) == RandomForestClassifier:
        res['description'] = str(zip(raw_data.columns, list(model.feature_importances)))
    else:
        raise RuntimeError, "Unknown model type"
    return res

#Declarative model training
MODEL_SPEC = {'logistic_regression' : LogisticRegression,
              'random_forest': RandomForestClassifier}

def train_model(raw_data, model_config):
    X = design_matrix(raw_data, model_config['feature_fns'])
    model = MODEL_SPEC[model_config['model_type']]()
    model.fit(X, labels(raw_data))
    model.description = model_description(model, model_config, raw_data)

    now_str = str(datetime.datetime.now()).replace(' ','_').replace('.','_').replace(':','_')
    with open('model_descriptions/%s.txt' % now_str, 'w') as f:
        f.write(str(model.description))

    with open('model.pkl', 'w') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    model_config = {
        'model_type': 'logistic_regression',
        'feature_fns': ['category_median_salary',
                        'contract_type_one_hot',
                        'contract_time_one_hot',
                        'num_low_salary_words',
                        'num_high_salary_words']
    }
    train_model(df, model_config)
