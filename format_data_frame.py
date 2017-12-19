def format_data_frame(df, response_column_label, print_verbose):

    list_of_column_labels = list(df)
    ordered_list_of_column_labels = list()
    for column_label in list_of_column_labels:
        if column_label == response_column_label:
            ordered_list_of_column_labels.append(column_label)
            list_of_column_labels.remove(column_label)
            break

    ordered_list_of_column_labels.extend(list_of_column_labels)
    df_formatted = df[ordered_list_of_column_labels].copy()

    if print_verbose:
        print()
        print('inside format_data_frame')
        print()
        print(df.head())
        print()
        print(response_column_label)
        print()
        print(df_formatted.head())

    return df_formatted
