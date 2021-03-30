import numpy as np
import csv
import igraph


def calc_dpck_connector(dpck, dpck_intra, dpck_flux):
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
    """Calculate the numerator for the probability of connectivity

    @:param p_matrix: an adjacency matrix containing the probabilities dispersal (potentially indirect) between patches
    @:param scores: the scores for each patch

    Calculate the numerator of the probability of connectivity metric (PC) defined by Saura and Rubio in
    A common currency for the different ways in which patches and links can contribute to habitat
    availability and connectivity in the landscape DOI:  https://doi.org/10.1111/j.1600-0587.2009.05760.x
    """
    numerator = np.sum(np.dot(scores, np.dot(scores.transpose(), p_matrix)))
    return numerator


def calc_pc(p_matrix, scores):
    """Calculate the probability of connectivity metric (PC)

        @:param p_matrix: an adjacency matrix containing the probabilities dispersal (potentially indirect) between patches
        @:param scores: the scores for each patch

        Calculate the probability of connectivity metric (PC) defined by Saura and Rubio in
        A common currency for the different ways in which patches and links can contribute to habitat
        availability and connectivity in the landscape DOI:  https://doi.org/10.1111/j.1600-0587.2009.05760.x
        """
    numerator = calc_pc_numerator(p_matrix, scores)
    return numerator/np.sum(scores)

def load_csv_data(colname, file_path):
    with open(file_path) as scores_file:
        reader = csv.DictReader(scores_file)
        scores_list = list()
        for row in reader:
            scores_list.append(float(row[colname]))
        scores_arr = np.array(scores_list)
        return scores_arr


def load_matrix(patch_count, file_path, indexing = 0):
    """ Load connectivity data from file into an Adjacency Matrix

    @:param patch_count: the expected number of patches in the file
    @:param file_path: the path to the file containing the data to be loaded

    Loads data from a file into a matrix. The input file must be a tab seperated file
    with the first patch number in the first column, the second patch number in the second column,
    and the probability of connectivity from the patch in the first column to the patch in the
    second column in the third column. Patch numbers should start at 1.
    """
    mat_file = open(file_path, "r")
    data = mat_file.readlines()
    mat_file.close()
    data_arr = np.zeros((patch_count, patch_count), dtype=np.float64)
    for line in data:
        i, j, x = str.split(line, "\t")
        if indexing == 0:
            if i == patch_count or j == patch_count:
                raise IndexError("Error, index equal to patch_count detected. Ensure data indices start at 0.")
            data_arr[int(i), int(j)] = x
        else:
            if i == 0 or j == 0:
                raise IndexError("Error, index equal to 0 detected. Ensure data indices start at 1.")
            data_arr[int(i) - 1, int(j) - 1] = x
    return data_arr


def calc_dpc_all(conn_data_path, scores_path, output_path, scores_colname, num_patches, start_patch, end_patch, adjust_scores = False, indexing = 0):
    # load data and set up
    connmat_reduced = load_matrix(num_patches, conn_data_path, indexing)
    scores = load_csv_data(scores_colname, scores_path)
    if adjust_scores:
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
    res_len = end_patch - start_patch
    dpck = np.zeros(res_len, dtype=np.float64)
    dpck_flux = np.zeros(res_len, dtype=np.float64)
    dpck_intra = np.zeros(res_len, dtype=np.float64)
    dpck_connector = np.zeros(res_len, dtype=np.float64)
    for i in range(start_patch, end_patch):
        g_removed_probs = make_removed(i, g)
        dpck[i - start_patch] = calc_dpck(i, g_removed_probs, scores, pc)
        dpck_flux[i - start_patch] = calc_dpck_flux(i, init_probs, scores, pc, num_patches)
        dpck_intra[i - start_patch] = calc_dpck_intra(i, scores, pc)
        dpck_connector[i - start_patch] = calc_dpck_connector(dpck[i - start_patch], dpck_intra[i - start_patch],
                                                              dpck_flux[i - start_patch])
        print("Done patch " + str(i))

    #TODO add lat and long to output
    print("Done calculating values")
    data_file = open(output_path, "w")
    print("Writing Values to File")
    # write values to file
    for i in range(start_patch, end_patch):
        data_file.write(str(i) + "\t" + str(dpck[i - start_patch]) + "\t" + str(dpck_intra[i - start_patch]) + "\t" +
                        str(dpck_flux[i - start_patch]) + "\t" + str(dpck_connector[i - start_patch]) + "\n")
    data_file.close()

if __name__ == "__main__":
    calc_dpc_all("../R_to_Py_connmat_reduced", "../scaled_scores_latlong_reproj_SJ.csv", "Patch_con_5_7000-9000",
                 "Avg_rescor", 12292, 7000, 9000, True, 1)