import numpy as np
import csv
import igraph
import math

def calc_dpck_connector(node_id, p_matrix, scores, dpck, dpck_intra, dpck_flux):
    dpck_connector = dpck - dpck_flux - dpck_intra

    return dpck_connector

def calc_dpck_flux(node_id, p_matrix, scores, init_pc, dims):
    row = p_matrix[node_id, :]
    col = p_matrix[:, node_id]
    # create a matrix with only row node_id and column node_id filled in, with the rest zeroes
    cross_matrix = np.zeros((dims, dims))
    cross_matrix[:, node_id] = col
    cross_matrix[node_id, :] = row
    cross_matrix[node_id, node_id] = 0
    dpck_flux = calc_pc_numerator(cross_matrix, scores)

    return (dpck_flux / init_pc) * 100

def calc_dpck_intra(node_id, scores, init_pc):
    d_pck_intra = scores[node_id]*scores[node_id]

    return (d_pck_intra / init_pc)*100

def calc_dpck(node_id, g_removed, scores, init_pc):
    save = scores[node_id]
    scores[node_id] = 0
    removed_pc = calc_pc_numerator(g_removed, scores)
    scores[node_id] = save
    dpck = 100*(init_pc - removed_pc)/init_pc

    return dpck

def make_removed(node_id, g):
    g_removed = g.copy()
    g_removed.delete_edges(g_removed.incident(vertex=node_id, mode=3))
    g_removed_probs = np.exp(-1 * np.array(g_removed.shortest_paths_dijkstra(weights='weight')))
    return g_removed_probs

def calc_pc_numerator(p_matrix, scores):
    numerator = np.sum(np.dot(scores, np.dot(scores.transpose(), p_matrix)))
    return numerator

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
def load_matrix(patch_count, file_path):
    mat_file = open("R_to_Py_connmat_reduced", "r")
    data = mat_file.readlines()
    mat_file.close()
    dim = math.sqrt(len(data))
    if (patch_count != dim):
        raise Exception("Number of rows does not equal the square of the number of patches")
    data_arr = np.zeros((dim, dim), dtype=np.float64)
    for line in data:
        i, j, x = str.split(line, "\t")
        data_arr[int(i) - 1, int(j) - 1] = x
    return data_arr

def calc_dpc_all(num_patches, start_patch, end_patch):
    # load data and set up
    connmat_reduced = load_matrix(num_patches)
    scores = load_scores()
    min_score = np.amin(scores)
    scores = scores + abs(min_score)

    # create igraph graph object from loaded adjacency matrix
    g = igraph.Graph.Adjacency((connmat_reduced > 0).tolist(), mode="DIRECTED")
    # Take the negative log of the edge probabilities
    g.es['weight'] = -1 * np.log(connmat_reduced[connmat_reduced.nonzero()])
    # find the highest probability path between patches using dijkstra with 1 on diag and then undoing -log operation
    init_probs = np.exp(-1 * np.array(g.shortest_paths_dijkstra(weights='weight')))

    # calculate the probability of connectivity for the landscape
    pc = calc_pc_numerator(init_probs, scores)

    # initialize arrays to store results
    range = end_patch - start_patch + 1
    dpck = np.zeros(range, dtype=np.float64)
    dpck_flux = np.zeros(range, dtype=np.float64)
    dpck_intra = np.zeros(range, dtype=np.float64)
    dpck_connector = np.zeros(range, dtype=np.float64)
    for i in range(start_patch, end_patch):
        g_removed_probs = make_removed(i, g)
        dpck[i] = calc_dpck(i, g_removed_probs, scores, pc)
        dpck_flux[i] = calc_dpck_flux(i, connmat_reduced, scores, pc, 12292)
        dpck_intra[i] = calc_dpck_intra(i, scores, pc)
        dpck_connector[i] = calc_dpck_connector(i, connmat_reduced, scores, dpck[i], dpck_intra[i], dpck_flux[i])
        print("Done patch " + str(i))

    print("Done calculating values")
    data_file = open("Patch_Con_Data_corrected_2_3999-7000", "w")
    print("Writing Values to File")
    # write values to file
    for i in range(start_patch, end_patch):
        data_file.write(str(i) + "\t" + str(dpck[i]) + "\t" + str(dpck_intra[i]) + "\t" +
                        str(dpck_flux[i]) + "\t" + str(dpck_connector[i]) + "\n")
    data_file.close()