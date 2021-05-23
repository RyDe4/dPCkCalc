import numpy as np
import csv
import igraph


def calc_dpck_connector(dpck, dpck_intra, dpck_flux):
    """Calculate the dPCk connector for a patch

    :param dpck: the dPCk for the patch in question
    :param dpck_intra: the dPCk intra for the patch in question
    :param dpck_flux: the dPCk flux for the patch in question
    :return: the dPCk connector for patch k

    Calculates the dPCk connector for patch k using dPCk connector = dPCk - dPCk flux - dPCk intra
    """
    dpck_connector = dpck - dpck_flux - dpck_intra

    return dpck_connector


def calc_dpck_flux(node_id, p_matrix, scores, init_pc, dims):
    """Calculate the dPCk flux for node k

    :param node_id: the node id(k) that dPCk flux will be calculated for
    :param p_matrix: a matrix of max probability path probabilities
    :param scores: a numpy array of habitat scores
    :param init_pc: the PC for the full network
    :param dims: the number of patches in p_matrix
    :return: returns the dPCk for k = node_id

    Calculates the dPCk flux for node k.
    """
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
    """Calculate the dPCK intra for node k

    :param node_id: the node id(k) that dPCk intra will be calculated for
    :param scores: a numpy array of habitat scores
    :param init_pc: the PC for the landscape before patch removal
    :return: returns the dPCk for k = node_id

    Calculates the dPCk intra for the node k(node_id) provided the patch scores and initial network PC
    """
    d_pck_intra = scores[node_id]*scores[node_id]

    return (d_pck_intra / init_pc)*100


def calc_dpck(node_id, g_removed, scores, init_pc):
    """Calculate the dPCk for node k

    :param node_id: the node id(k) that dPCk will be calculated for
    :param g_removed: an adjacency matrix like matrix of the maximum probabilies of optimal path between patches in the
    network after the removal of node k(node id)
    :param scores: a numpy array of habitat scores
    :param init_pc: the PC for the landscape before patch removal
    :return: returns the dPCk for k = node_id

    Calculates the dPCk for the node k(node_id), provided the maximum probability paths in the network after removal of
    patch k, the habitat quality scores, and the initial PC of the full network.
    """
    save = scores[node_id]
    scores[node_id] = 0
    removed_pc = calc_pc_numerator(g_removed, scores)
    scores[node_id] = save
    dpck = 100*(init_pc - removed_pc)/init_pc

    return dpck


def make_removed(node_id, g):
    """ Recalculate the shortest paths in the graph and convert them to probabilities after removing a node

    :param node_id: the node id of the node to remove
    :param g: a graph where the edge weights are the -log probabilities of the probabilities of direct dispersal
    :return: a adjacency matrix-like matrix where the entries are the maximum probabilities for the optimal path between
    the patches.

    Recalculate the shortest paths in the graph and convert them to probabilities after removing a node.
    """
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
    if p_matrix.shape[0] != len(scores):
        raise Exception("matrix dimensions must be equal to length of scores vector")
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
    if p_matrix.shape[0] != len(scores):
        raise Exception("matrix dimensions must be equal to length of scores vector")
    numerator = calc_pc_numerator(p_matrix, scores)
    return numerator/np.sum(scores)


def load_csv_data(colname, file_path):
    """Load a csv column into a numpy array

    :param colname: name of the column in the csv file
    :param file_path: path to the csv file
    :return: a numpy array containing the data in the column "colname" in the file at file_path

    Creates a numpy array from a column of a csv file
    """
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
    """ Calculate dPCk, intra, flux, and connector, and write these metrics to file

        :param conn_data_path: path to the file containing connectivity data. This should be a tab separated file with the
        first and second columns being patch ids
        :param scores_path: path to the file containing habitat score data. This should be a csv file with scores in column
        with a column name
        :param output_path: the path to the output file where the results will be written
        :param scores_colname: the name of the scores column in the scores_path csv file
        :param num_patches: the number of habitat patches in the input files
        :param start_patch: the start patch for metric calculation. Overall PC is calculated for all patches, regardless
        of the value for this argument
        :param end_patch: the end patch for metric calculation. Overall PC is calculated for all patches, regardless
        of the value for this argument
        :param adjust_scores: if True, add the minimum habitat score to all the scores. Defualt is false.
        :param indexing: does indexing in input files start at 0 or 1? Defualt is 0

        Calculates dPCk, intra, flux, and connector, and write theses metrics to file. The output file is a tab delimited
        file with patch number/id in the first column , dPCk in the second column, intra in the 3rd column, flux in the
        4th column, and connector in the 5th column.
    """
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

    print("Done calculating values")
    data_file = open(output_path, "w")
    print("Writing Values to File")
    # write values to file
    for i in range(start_patch, end_patch):
        data_file.write(str(i) + "\t" + str(dpck[i - start_patch]) + "\t" + str(dpck_intra[i - start_patch]) + "\t" +
                        str(dpck_flux[i - start_patch]) + "\t" + str(dpck_connector[i - start_patch]) + "\n")
    data_file.close()

