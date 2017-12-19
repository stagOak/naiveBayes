import pandas as pd
import numpy as np
from naiveBayes import NaiveBayes

from format_data_frame import format_data_frame


def naive_bayes_glue_layer(df_train, df_test, response_column_label):

    # print('inside naive_bayes_glue_layer')

    df_train = format_data_frame(df_train, response_column_label, print_verbose=False)
    df_test = format_data_frame(df_test, response_column_label, print_verbose=False)

    # instantiate naive bayes classifier object then fit and predict
    nb = NaiveBayes()
    nb.get_data(df_train)
    nb.fit()
    prediction_prob, prediction, accuracy_score = nb.predict(df_test)

    # return probability, classification
    return prediction_prob, prediction, accuracy_score
