import csv
import pandas as pd


def import_csv(filename, dataformat="dataframe") -> tuple:
    """
    Imports a classification .csv with data in the first columns and the label in the final column
    :param filename: path to the file
    :param dataformat: "lists" or "dataframe" -> return type
    :return: tuple(data, labels) where data is a list of list containing the evidence and labels is a list with all labels
    """
    if dataformat == "lists":
        data, labels = [], []
        # open csv file
        with open(filename) as f:
            #as csv's contain labels use DictReader
            reader = csv.DictReader(f, delimiter=';')
            headers = reader.fieldnames
            for row in reader:
                data.append([int(row[value]) for value in row][:-1])
                labels.append(1 if row[headers[-1]] == '1' else 0)
        return data, labels
    return pd.read_csv(filename, sep=";")


if __name__ == "__main__":
    import_csv("Data/divorce.csv")
