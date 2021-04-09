import igraph
from PcCalc import load_matrix
from PcCalc import load_csv_data
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

def load_pc_data(data_len, file_path):
    mat_file = open(file_path, "r")
    data = mat_file.readlines()
    mat_file.close()
    data_arr = np.zeros((data_len, 5), dtype=np.float64)
    for line in data:
        i, dpck, dpck_intra, dpck_flux, dpck_connector = str.split(line, "\t")
        data_arr[int(i)][0] = i
        data_arr[int(i)][1] = dpck
        data_arr[int(i)][2] = dpck_intra
        data_arr[int(i)][3] = dpck_flux
        data_arr[int(i)][4] = dpck_connector
    return data_arr

def pie_composition(dpck_data, intra_data, flux_data, connector_data, title, percentile = None, top_percent = True):
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
    # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
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

