import pandas as pd
import numpy as np


def load_data_split_train_test(file_path_and_name, train_split_fraction, set_seed, print_verbose):

    df = pd.read_csv(file_path_and_name)

    # shuffle the data frame
    if set_seed:
        np.random.seed(0)

    df = df.sample(frac=1).reset_index(drop=True)

    num_cols = df._get_numeric_data().columns
    if len(num_cols) > 0:
        raise ValueError('All data must be categorical.')

    if set_seed:
        np.random.seed(0)

    msk = np.random.rand(len(df)) < train_split_fraction
    df_train = df[msk].reset_index(drop=True)
    df_test = df[~msk].reset_index(drop=True)

    if print_verbose:
        print('\n*******')
        print('data frame loaded (df):')
        print(df.head())
        print('\n*******')
        print('df.shape = ', df.shape)
        print('\n*******')
        print('df_train.shape = ', df_train.shape)
        print('\n*******')
        print('df_test.shape = ', df_test.shape)

    return df_train, df_test
