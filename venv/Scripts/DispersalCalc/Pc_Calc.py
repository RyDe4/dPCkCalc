import numpy as np
import csv

# TODO define standard input file descriptions
def load_scores(colname):
    with open("scaled_scores_latlong_reproj_SJ.csv") as scores_file:
        reader = csv.DictReader(scores_file)
        scores_list = list()
        for row in reader:
            scores_list.append(float(row[colname]))
        scores_arr = np.array(scores_list)
        return scores_arr

# TODO define standard input file descriptions
def load_matrix():
    mat_file = open("R_to_Py_connmat_reduced", "r")
    data = mat_file.readlines()
    mat_file.close()
    data_arr = np.zeros((12292, 12292), dtype=np.float64)
    for line in data:
        i, j, x = str.split(line, "\t")
        data_arr[int(i) - 1, int(j) - 1] = x
    return data_arr



