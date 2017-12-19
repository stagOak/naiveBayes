class NaiveBayes:

    def __init__(self):
        import pandas as pd
        import numpy as np
        self.id = id
        self._num_obs = 0
        self._df = pd.DataFrame()
        self._df_fit = pd.DataFrame()
        self._df_test = pd.DataFrame()
        self._fit_dict_row_counts_dict = dict()
        self._fit_response_count_dict = dict()
        self._fit_response_level_list = list()
        self._fit_response_level_priors_dict = dict()
        self._fit_df_column_labels_list = list()
        self._predict_dict = dict()
        self._fit_flag = False
        self.test_truth = np.array([])
        self._prediction = np.array([])
        self.accuracy_score = np.nan
        self._prediction_prob = np.array([])

    def get_data(self, df):
        self._df = df
        self._num_obs = self._df.shape[0]
        self._fit_df_column_labels_list = list(self._df.columns)
        for column in self._df.columns:
            self._df[column] = self._df[column].astype('category')
        return self._df

    def get_priors(self):
        from collections import Counter
        temp_list = self._df.loc[:, self._fit_df_column_labels_list[0]].tolist()
        temp_counter = Counter(temp_list)
        for response_level in self._fit_response_level_list:
            self._fit_response_level_priors_dict[response_level] = temp_counter[response_level]/self._num_obs

    def _set_basic_counts(self, list_of_outcomes):
        for outcome in list_of_outcomes:
            num_rows_count, response_count_dict = self._get_counts(outcome)
            self._fit_dict_row_counts_dict[outcome] = num_rows_count
            self._fit_response_count_dict[outcome] = response_count_dict[outcome]
        self.get_priors()

    def _fill_in_the_basics(self, row_num, df_fit_columns, outcome):
        row_num += 1
        feature_num = -1
        for feature in self._fit_df_column_labels_list[1:]:
            feature_num += 1
            # print(feature, df_fit_columns[feature_num])
            assert(feature == df_fit_columns[feature_num])
            self._df_fit.at[row_num, feature] = outcome[feature_num].split('::')[1]

        feature_num += 1
        self._df_fit.at[row_num, df_fit_columns[feature_num]] = self._fit_dict_row_counts_dict[outcome]
        feature_num += 1
        self._df_fit.at[row_num, df_fit_columns[feature_num]] = \
            self._fit_response_count_dict[outcome][df_fit_columns[feature_num]]
        feature_num += 1
        self._df_fit.at[row_num, df_fit_columns[feature_num]] = \
            self._fit_response_count_dict[outcome][df_fit_columns[feature_num]]
        return row_num, feature_num

    def _get_cond_prob(self, column_label, outcome, row_num, df_fit_columns, feature_num):
        temp = column_label.split('|')
        temp_feature_column_label = temp[0]
        temp_response_label_level = temp[1]
        for element in outcome:
            if temp_feature_column_label in element:
                temp_level = element.split('::')[1]
                df_temp = self._df[self._df.loc[:, temp_feature_column_label] == temp_level].copy()
                df_temp = df_temp[df_temp.loc[:, self._fit_df_column_labels_list[0]] ==
                                  temp_response_label_level].copy()
                prob_feat_level_and_resp_level = df_temp.shape[0] / self._num_obs
                prob_resp_level = self._fit_response_level_priors_dict[temp_response_label_level]
                prob_feat_level_given_resp_level = prob_feat_level_and_resp_level / prob_resp_level
                self._df_fit.at[row_num, column_label] = prob_feat_level_given_resp_level
                break

    def _get_score(self, resp_level, df_fit_columns, row_num, current_column_label):
        pattern1 = '|' + resp_level
        pattern2 = 'prior ' + resp_level
        score = 1
        for column_label in df_fit_columns:
            if pattern1 in column_label.lower() or pattern2 in column_label.lower():
                score *= self._df_fit.at[row_num, column_label]
        self._df_fit.at[row_num, current_column_label] = score

    def _get_prob(self, df_fit_columns, resp_level, current_column_label, row_num):
        sum_score = 0
        for level in self._fit_response_level_list:
            for column_label in df_fit_columns:
                if 'score' in column_label and level in column_label:
                    sum_score += self._df_fit.at[row_num, column_label]
                    if level.lower() == resp_level.lower():
                        resp_level_score = self._df_fit.at[row_num, column_label]

        # ToDo: avoid division by zeroe here
        self._df_fit.at[row_num, current_column_label] = resp_level_score/sum_score

    def _get_most_prob(self, df_fit_columns, row_num, current_column_label):
        find_most_prob_dict = dict()
        for level in self._fit_response_level_list:
            for column_label in df_fit_columns:
                if 'prob' in column_label.lower() and level.lower() in column_label.lower():
                    find_most_prob_dict[level] = self._df_fit.at[row_num, column_label]
        self._df_fit.at[row_num, current_column_label] = \
            max(find_most_prob_dict.keys(), key=(lambda k: find_most_prob_dict[k]))

    def _fill_in_the_rest(self, row_num, df_fit_columns, outcome, feature_num):
        for column_label in df_fit_columns[feature_num:]:
            if '|' in column_label:
                self._get_cond_prob(column_label, outcome, row_num, df_fit_columns, feature_num)
            elif 'prior' in column_label.lower():
                response_level = column_label.split(' ')[1]
                self._df_fit.at[row_num, column_label] = self._fit_response_level_priors_dict[response_level]
            elif 'score' in column_label.lower():
                resp_level = column_label.split(' ')[1].lower()
                self._get_score(resp_level, df_fit_columns, row_num, column_label)
            elif 'prob' in column_label.lower() and 'most' not in column_label.lower():
                resp_level = column_label.split(' ')[1].lower()
                self._get_prob(df_fit_columns, resp_level, column_label, row_num)
            elif 'most prob' in column_label.lower():
                self._get_most_prob(df_fit_columns, row_num, column_label)
            else:
                quit('error: column_label unrecognized')

    def fit(self):
        self._fit_flag = True
        list_of_outcomes = self._set_starting_fit_dict()
        self._set_basic_counts(list_of_outcomes)
        self._df_fit = self._create_empty_fit_df()
        row_num = -1
        df_fit_columns = list(self._df_fit.columns)
        for outcome in list_of_outcomes:
            row_num, feature_num = self._fill_in_the_basics(row_num, df_fit_columns, outcome)
            feature_num += 1
            self._fill_in_the_rest(row_num, df_fit_columns, outcome, feature_num)

    def _create_empty_fit_df(self):
        import pandas as pd
        # create the fit data frame
        starting_df_fit_column_labels_list = \
            [self._fit_df_column_labels_list[1:], ['N_Rows'], self._fit_response_level_list]
        flat_starting_df_fit_column_labels_list = \
            [item for sublist in starting_df_fit_column_labels_list for item in sublist]

        for response_level in self._fit_response_level_list:
            for feature in self._fit_df_column_labels_list[1:]:
                flat_starting_df_fit_column_labels_list.append(feature + '|' + response_level)
            flat_starting_df_fit_column_labels_list.append('prior ' + response_level)
            flat_starting_df_fit_column_labels_list.append('score ' + response_level)

        for response_level in self._fit_response_level_list:
            flat_starting_df_fit_column_labels_list.append('Prob ' + response_level)

        flat_starting_df_fit_column_labels_list.append('most Prob ' + self._fit_df_column_labels_list[0])

        df_fit = pd.DataFrame(columns=flat_starting_df_fit_column_labels_list)
        return df_fit

    def _get_counts(self, outcome):
        # counts the number of times an outcome occurs in the data frame
        # gets a frequency distribution for the response levels for the outcome
        # print('inside _get_counts')
        df = self._df
        column_number = 0
        for element in outcome:
            temp_element = element.split('::')[1]
            column_number += 1
            df = df.loc[df[df.columns[column_number]] == temp_element].copy()
        num_rows_count = df.shape[0]
        response_value_count = df[df.columns[0]].value_counts()
        response_count_dict = dict()
        response_count_dict[outcome] = dict()
        for index, count in response_value_count.items():
            response_count_dict[outcome][index] = count
            if index not in self._fit_response_level_list:
                self._fit_response_level_list.append(index)
        # print('leaving _get_counts')
        return num_rows_count, response_count_dict

    def _set_starting_fit_dict(self):
        # gets a list of tuples which enumerates all possible outcomes given the features and the levels in data frame
        # if there are nx features then a tuple will be nx dimensions
        # each element in the tuple has feature column label and feature level separatered by double colon
        import itertools
        feature_dict = dict()
        for row in self._df.itertuples(index=False, name=''):
            column_number = 0
            for element in row[1:]:
                column_number += 1
                key = self._df.columns[column_number]
                if key in feature_dict:
                    feature_dict[key].append(key + '::' + element)
                else:
                    feature_dict[key] = list([key + '::' + element])
        feature_list = list()
        for key, value in feature_dict.items():
            feature_list.append(list(set(value)))
        list_of_outcomes = list(itertools.product(*feature_list))
        return list_of_outcomes

    def predict(self, df):
        import numpy as np
        if not self._fit_flag:
            quit('cannot predict - no model has been fit - please run fit')
        # print('inside predict')
        self._df_test = df
        self._prediction = np.empty((self._df_test.shape[0], 1), dtype=object)
        self._prediction[:] = 'blank'
        self.test_truth = np.empty((self._df_test.shape[0], 1), dtype=object)
        self.test_truth[:] = 'blank'
        self._prediction_prob = np.empty((self._df_test.shape[0], 1), dtype=float)
        self._prediction[:] = np.nan
        row_index = -1
        count = 0
        score = 0
        column_list = self._df_fit.columns
        num_columns = len(column_list)
        for row in self._df_test.itertuples(index=True, name='Pandas'):
            count += 1
            row_index += 1
            column = -1
            df_temp = self._df_fit.copy()
            for label in self._fit_df_column_labels_list:
                column += 1
                if column == 0:
                    the_truth = getattr(row, label)
                else:
                    df_temp = df_temp[df_temp[label] == getattr(row, label)]

            df_temp = df_temp.reset_index(drop=True)

            self._prediction[row_index] = df_temp.iloc[0][num_columns - 1]
            if self._prediction[row_index][0] in column_list[num_columns - 3]:
                self._prediction_prob[row_index] = df_temp.iloc[0][num_columns - 3]
            elif self._prediction[row_index][0] in column_list[num_columns - 2]:
                self._prediction_prob[row_index] = df_temp.iloc[0][num_columns - 2]
            else:
                quit('error - probability for class predicted not found')

            self.test_truth[row_index] = the_truth
            if self._prediction[row_index] == self.test_truth[row_index]:
                score += 1

        self.accuracy_score = score/count

        return self._prediction_prob, self._prediction, self.accuracy_score
