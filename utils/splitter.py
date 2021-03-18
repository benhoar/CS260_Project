import csv

OUTPUT_FILE1 = 'split_categorical.csv'
OUTPUT_FILE2 = 'split_numerical.csv'

with open('Stocks_forML_Feb24.csv', mode='r') as in_file:
    csv_reader = csv.reader(in_file)
    next(csv_reader)
    categorical_inds = []
    for row in csv_reader:
        for idx, item in enumerate(row):
            if item == "True" or item == "False":
                categorical_inds.append(idx)
        break

    in_file.seek(0)
    new_reader = csv.reader(in_file)
    with open(OUTPUT_FILE1, "w") as out_file1:
        categorical_writer = csv.writer(out_file1)
        for row in new_reader:
            categorical_row = []
            for idx, item in enumerate(row):
                if idx in categorical_inds:
                    categorical_row.append(item)
            categorical_writer.writerow([row[0]] + categorical_row + [row[-1]])

    in_file.seek(0)
    new_reader = csv.reader(in_file)
    with open(OUTPUT_FILE2, "w") as out_file2:
        numerical_writer = csv.writer(out_file2)
        for row in new_reader:
            numerical_row = []
            for idx, item in enumerate(row):
                if idx not in categorical_inds:
                    numerical_row.append(item)
            numerical_writer.writerow(numerical_row)