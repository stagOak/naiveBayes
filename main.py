from naive_bayes_glue_layer import naive_bayes_glue_layer
from load_data_split_train_test import load_data_split_train_test
from sklearn.metrics import confusion_matrix


# load data and get train and test sets

file_path_and_name = 'miniTennis.csv'
classification_variable = 'y'

# file_path_and_name = 'Titanic.csv'
# classification_variable = 'Survived'

df_train, df_test = load_data_split_train_test(file_path_and_name, train_split_fraction=0.7, set_seed=True,
                                               print_verbose=False)

# in this assignment the Naive Bayes classifier is implemented as a class called NaiveBayes
# the naive_bayes_glue_layer below bridges the assignment requirements and the object oriented design

probability, classification, accuracy_score = naive_bayes_glue_layer(df_train, df_test, classification_variable)

print()
print('classification prediction:')
print(classification)
print()
print('truth:')
print(df_test.loc[:, classification_variable])
print()
print('classification probability:')
print(probability)
print()
print('accuracy_score = ', accuracy_score)
print()
print('confusion matrix: rows are truth, columns are prediction:')
print(confusion_matrix(df_test.loc[:, classification_variable], classification))
