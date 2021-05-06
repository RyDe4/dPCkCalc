import igraph
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import shapely.geometry
import numpy as np
from descartes.patch import PolygonPatch
import matplotlib
import seaborn as sb
import math
import statistics

def make_igraph_graph(p_matrix):
    """Create a directed igraph graph object from an numpy adjacency matrix that could contain zeroes.

    :param p_matrix: a 2D numpy array representing a graph in adjacency matrix form
    :return: an igraph graph
    """
    g = igraph.Graph.Adjacency((p_matrix > 0).tolist(), mode="DIRECTED")
    return g

def plot_smoothed_density_all(dpck_data, intra_data, flux_data, connector_data, titles):
    hist_kw = {"bins" : 100}
    sb.displot(dpck_data, kde=True, kind="hist", **hist_kw)
    plt.title(titles[0])
    plt.subplots_adjust(top=0.945, bottom=0.102, left=0.14, right=0.97, hspace=0.2, wspace=0.2)
    plt.show()
    sb.displot(intra_data, kde=True, kind="hist", **hist_kw)
    plt.title(titles[1])
    plt.subplots_adjust(top=0.945, bottom=0.102, left=0.14, right=0.97, hspace=0.2, wspace=0.2)
    plt.show()
    sb.displot(flux_data, kde=True, kind="hist", **hist_kw)
    plt.title(titles[2])
    plt.subplots_adjust(top=0.945, bottom=0.102, left=0.14, right=0.97, hspace=0.2, wspace=0.2)
    plt.show()
    sb.displot(connector_data, kde=True, kind="hist", **hist_kw)
    plt.title(titles[3])
    plt.subplots_adjust(top=0.945, bottom=0.102, left=0.14, right=0.97, hspace=0.2, wspace=0.2)
    plt.show()
    sb.kdeplot(data={"dpck": dpck_data, "dpck_intra": intra_data, "dpck_flux": flux_data, "dpck_connector": connector_data})
    plt.subplots_adjust(top=0.945, bottom=0.102, left=0.122, right=0.97, hspace=0.2, wspace=0.2)
    plt.show()

def load_pc_data(data_len, file_path, coords =False):
    """Load PC data into a numpy array

    :param data_len: the number of patches in the input file
    :param file_path: the path to the file with the connectivity data. Should be in the format output by
     PcCalc.calc_dpc_all
    :param coords: True if the last to columns of input file contain latitude and longitude coordinates. False by
    defualt.
    :return: a 2D numpy ndarray with column 0 representing the patch numbers, column 1 containing the dPCk for the patch
    indicated in column 0, column 2 is intra, column 3 is flux, column 4 is connector
    """
    mat_file = open(file_path, "r")
    data = mat_file.readlines()
    mat_file.close()
    data_arr = np.zeros((data_len, 5), dtype=np.float64)
    for line in data:
        if coords == False:
            i, dpck, dpck_intra, dpck_flux, dpck_connector = str.split(line, "\t")
        else:
            i, dpck, dpck_intra, dpck_flux, dpck_connector, lat, long = str.split(line, "\t")
        data_arr[int(i)][0] = i
        data_arr[int(i)][1] = dpck
        data_arr[int(i)][2] = dpck_intra
        data_arr[int(i)][3] = dpck_flux
        data_arr[int(i)][4] = dpck_connector
    return data_arr

def pie_composition(dpck_data, intra_data, flux_data, connector_data, title, percentile = None, top_percent = True):
    """Create a pie chart of the contribution of each dPCk fraction

    :param dpck_data: numpy array of dPCk data
    :param intra_data: numpy array of dPCk intra data
    :param flux_data: numpy array of dPCk flux data
    :param connector_data: numpy array of dPCk connector data
    :param title: String indicating the title for the pie chart
    :param percentile: the top or bottom percentile for the pie chart
    :param top_percent: if True, percentile indicates the top "percentile" percentile. If false, percentile indicates
    the bottom "percentile" percentile. Defaults to True.

    Create a pie chart of the average contribution of each dPCk fraction to the total dPCk for each patch. The
    percentile and top_percentile parameters indicate if only patches in a top percentile or bottom percentile should
    be considered.
    """
    intra_total = 0
    flux_total = 0

    #sort the data
    indices = np.argsort(dpck_data)
    dpck_data = dpck_data[indices]
    intra_data = intra_data[indices]
    flux_data = flux_data[indices]

    if percentile is not None:
        keep = math.ceil(percentile / 100 * len(dpck_data))
        if not top_percent:
            dpck_data_trimmed = dpck_data[0:keep]
            intra_data_trimmed = intra_data[0:keep]
            flux_data_trimmed = flux_data[0:keep]
        else:
            dpck_data_trimmed = dpck_data[len(dpck_data) - keep:len(dpck_data)]
            intra_data_trimmed = intra_data[len(dpck_data) - keep:len(dpck_data)]
            flux_data_trimmed = flux_data[len(dpck_data) - keep:len(dpck_data)]
    else:
        dpck_data_trimmed = dpck_data
        intra_data_trimmed = intra_data
        flux_data_trimmed = flux_data
    for i in range(len(dpck_data_trimmed)):
        if dpck_data_trimmed[i] > 0:
            intra_total = intra_total + intra_data_trimmed[i]/dpck_data_trimmed[i]
            flux_total = flux_total + flux_data_trimmed[i]/dpck_data_trimmed[i]
    intra_percent = intra_total/len(intra_data_trimmed)
    flux_percent = flux_total/len(flux_data_trimmed)
    connector_percent = 1 - flux_percent - intra_percent
    data_arr = np.array([intra_percent, flux_percent, connector_percent])
    labels = ["intra", "flux", "connector"]

    plt.pie(data_arr, labels=labels, autopct='%1.1f%%')
    plt.title(title)
    plt.show()

def graph_map(basemap_path, pc_data, g, lats, longs, x_left, x_right, y_bottom, y_top, scale_factor = 1, arrow_scale = 1,
              bound_poly = None, value_cap = False):
    """Visualize patches on a map, with patches represented as coloured points and dispersal represented by arrows

    :param basemap_path: Path to the basemap file. File should be dbf or any other format readable by the GeoPandas
    read_file() function
    :param pc_data: a numpy array containing the data values for each path
    :param g:
    :param lats: a numpy array of the latitudes of each patch, with the value at each indices corresponding to the value
    at the same indice of the pc_data array
    :param longs: a numpy array of the longitudes of each patch, with the value at each indices corresponding to the value
    at the same indice of the pc_data array
    :param x_left: the left longitude bound on the map's view
    :param x_right: the right longitude bound on the map's view
    :param y_bottom: the bottom latitude bound on the map's view
    :param y_top: the top latitude bound on the map's view
    :param scale_factor: positive int indicating the size of the patch points, with a larger number
    corresponding to a larger point
    :param arrow_scale: positive int, with larger values indicating thicker arrows
    :param bound_poly: a shapely polygon indicating which patches should be shown. Only patches within the polygon
    are shown
    :param value_cap: double, scales all pc_data values larger than value_cap to be equal to value cap. Can help with
    colourmap issues if the is a small number of patches with very large pc_data values.

    Create a map showing patches as small circles, with patches coloured based on some data value, and arrows indicating
    dispersal between patches.
    """
    # use subplots to add points on top of world map
    fig, ax = plt.subplots(1)

    # shrink all values greater than cap
    if value_cap:
        display_data = pc_data.copy()
        display_data[np.argwhere(pc_data > value_cap)] = value_cap
    else:
        display_data = pc_data

    # create a colourmap scaled by the given metric
    cmap = plt.get_cmap('hot_r')
    color_map = matplotlib.cm.ScalarMappable(cmap=cmap)
    colors = color_map.to_rgba(x=display_data)

    # restrict view to geographic bounds
    plt.axis([x_left, x_right, y_bottom, y_top])
    shrinkage_factor = min(scale_factor*(abs(x_right - x_left) / 360), scale_factor*(abs(y_top - y_bottom) / 180))

    # generate point polygons to plot
    points = []
    in_scope = []
    for i in range(len(pc_data)):
        point = shapely.geometry.Point(longs[i], lats[i])
        if bound_poly is not None and not point.within(bound_poly):
            continue
        circle = point.buffer(0.35 * shrinkage_factor)
        patch = PolygonPatch(circle, facecolor=colors[i], edgecolor=colors[i], alpha=0.7, zorder=2)
        points.append(patch)
        in_scope.append(i)

    # put the points in a collection and plot them
    collection = PatchCollection(points)
    ax.add_collection(collection)
    collection.set_color(colors)
    world = gpd.read_file(basemap_path)
    world.plot(ax=ax, facecolor='green')
    plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), ax=ax)

    #create arcs
    style = "Simple, tail_width=0.0001, head_width=" + str(1*arrow_scale) + ", head_length=" + str(2*arrow_scale)
    arrow_kw = dict(arrowstyle=style, color="black", lw=0.1)

    for i in range(len(display_data)):
        if bound_poly is not None and x not in in_scope:
            continue
        #get out neighbours
        out_neighbours = g.neighbors(i, mode = "out")
        #get in neighbours
        for x in out_neighbours:
            if bound_poly is not None and x not in in_scope:
                continue
            ax.add_patch(matplotlib.patches.FancyArrowPatch((longs[i], lats[i]),
                                                            (longs[x], lats[x]),
                                                                   connectionstyle="arc3,rad=-.5", **arrow_kw))


    plt.show()

def map_visualize(basemap_path, pc_data, lats, longs, x_left, x_right, y_bottom, y_top, scale_factor, polygon_bound = None,
                  plot_poly = False, value_cap = False):
    '''Visualize patches on a map, with patches represented as coloured points

    :param basemap_path: Path to the basemap file. File should be dbf or any other format readable by the GeoPandas
    read_file() function
    :param pc_data: a numpy array containing the data values for each path
    :param lats: a numpy array of the latitudes of each patch, with the value at each indices corresponding to the value
    at the same indice of the pc_data array
    :param longs: a numpy array of the longitudes of each patch, with the value at each indices corresponding to the value
    at the same indice of the pc_data array
    :param x_left: the left longitude bound on the map's view
    :param x_right: the right longitude bound on the map's view
    :param y_bottom: the bottom latitude bound on the map's view
    :param y_top: the top latitude bound on the map's view
    :param scale_factor: the scale factor for the size of the patch points, with a larger number corresponding to a
    larger point
    :param polygon_bound: a shapely polygon indicating which patches should be shown. Only patches within the polygon
    are shown
    :param plot_poly: boolean, if true the polygon_bound is shown on the map
    :param value_cap: double, scales all pc_data values larger than value_cap to be equal to value cap. Can help with
    colourmap issues if the is a small number of patches with very large pc_data values.

    Create a map showing patches as small circles, with patches coloured based on some data value.
    '''
    #use subplots to add points on top of world map
    fig, ax = plt.subplots(1)

    #shrink all values greater than cap
    if value_cap:
        display_data = pc_data.copy()
        display_data[np.argwhere(pc_data > value_cap)] = value_cap

    #create a colourmap scaled by the given metric
    cmap = plt.get_cmap('hot_r')
    color_map = matplotlib.cm.ScalarMappable(cmap=cmap)
    colors = color_map.to_rgba(x=display_data)

    #restrict view to geographic bounds
    plt.axis([x_left, x_right, y_bottom, y_top])
    shrinkage_factor = min(scale_factor*(abs(x_right - x_left)/360), scale_factor*(abs(y_top - y_bottom)/180))

    # generate point polygons to plot
    points = []

    for i in range(len(pc_data)):
        point = shapely.geometry.Point(longs[i], lats[i])
        if polygon_bound is not None and not point.within(polygon_bound):
            continue
        circle = point.buffer(0.35*shrinkage_factor)
        patch = PolygonPatch(circle, facecolor=colors[i], edgecolor=colors[i], alpha=0.7, zorder=2)
        points.append(patch)
    #put the points in a collection and plot them
    collection = PatchCollection(points)
    ax.add_collection(collection)
    collection.set_color(colors)
    world = gpd.read_file(basemap_path)
    world.plot(ax=ax, facecolor='green')
    if plot_poly:
        ax.add_patch(PolygonPatch(polygon_bound, alpha=0.1))
    plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), ax=ax)
    plt.show()

def get_in_bounds(data, lats, longs, polygon):
    data_keep = []
    lats_keep = []
    longs_keep = []
    for i in range(len(data)):
        if shapely.geometry.Point(longs[i], lats[i]).within(polygon):
            data_keep.append(data[i])
            lats_keep.append(lats[i])
            longs_keep.append(longs[i])

    return np.array(data_keep), np.array(lats_keep), np.array(longs_keep)

def avg_dist_edge(g, patch_id, lats, longs):
    out_neighbours = g.neighbors(patch_id, mode="out")
    in_neighbours = g.neighbors(patch_id, mode="in")
    if len(out_neighbours) == 0 or len(in_neighbours) == 0:
        return 0
    distances = []
    for i in in_neighbours:
        dist = math.sqrt(abs(lats[patch_id] - lats[i]) + abs(longs[patch_id] - longs[i]))
        distances.append(dist)
    for i in out_neighbours:
        dist = math.sqrt(abs(lats[patch_id] - lats[i]) + abs(longs[patch_id] - longs[i]))
        distances.append(dist)

    return statistics.mean(distances)

def scatter_edge_dist(g, data, lats, longs, title, x_lab, y_lab):
    x = np.zeros((len(data)))
    y = np.zeros((len(data)))
    for i in range(len(data)):
        x[i] = (data[i])
        y[i] = (avg_dist_edge(g, i, lats, longs)*111)
    indices = np.argwhere(y > 0)
    x = x[indices]
    y = y[indices]
    sb.regplot(x=x, y=y)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.show()

