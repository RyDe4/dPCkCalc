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


def make_igraph_graph(p_matrix):
    g = igraph.Graph.Adjacency((p_matrix > 0).tolist(), mode="DIRECTED")
    return g


def load_pc_data(file_path):
    mat_file = open(file_path, "r")
    data = mat_file.readlines()
    mat_file.close()
    data_arr = np.zeros((12292, 5), dtype=np.float64)
    for line in data:
        i, dpck, dpck_intra, dpck_flux, dpck_connector = str.split(line, "\t")
        data_arr[int(i)][0] = i
        data_arr[int(i)][1] = dpck
        data_arr[int(i)][2] = dpck_intra
        data_arr[int(i)][3] = dpck_flux
        data_arr[int(i)][4] = dpck_connector
    return data_arr


def calc_centroid(lats, longs):
    x = sum(lats)/len(lats)
    y = sum(longs)/len(longs)
    return x, y


def graph_vis(igraph_g, num_nodes, img_dims = 400, spatial = False, coord_long = None, coord_lat = None):
    igraph_g.vs['label'] = range(0, num_nodes)
    visual_style = {}
    out_name = "graph.png"
    visual_style["bbox"] = (img_dims, 2*img_dims)
    visual_style["margin"] = 0
    visual_style["vertex_color"] = 'green'
    if (spatial == True):
        for long in coord_long:
            if long > 180:
                long = long - 180
        layout = [(coord_long[i], coord_lat[i] + 90)for i in range(len(coord_long))]
        visual_style["layout"] = layout
    if (num_nodes < 50):
        visual_style["vertex_size"] = 15
        visual_style["vertex_label_size"] = 10
        visual_style["edge_label_size"] = 10
    elif (num_nodes < 200):
        visual_style["vertex_size"] = 13
        visual_style["vertex_label_size"] = 8
        visual_style["edge_label_size"] = 8
    else:
        visual_style["vertex_size"] = 12
        visual_style["vertex_label_size"] = 7
        visual_style["edge_label_size"] = 7
    visual_style["edge_curved"] = True

    igraph.plot(igraph_g, out_name, **visual_style)


def map_visualize(pc_data, lats, longs, x_left, x_right, y_bottom, y_top):
    #use subplots to add points on top of world map
    fig, ax = plt.subplots(1)

    #create a colourmap scaled by the given metric
    cmap = plt.get_cmap('hot_r')
    color_map = matplotlib.cm.ScalarMappable(cmap=cmap)
    #TODO modify multiplier to be more general
    colors = color_map.to_rgba(x=pc_data)


    #restrict view to geographic bounds
    plt.axis([x_left, x_right, y_bottom, y_top])
    shrinkage_factor = min(abs(x_right - x_left)/360, abs(y_top - y_bottom)/180)

    # generate point polygons to plot
    points = []
    #TODO move coordinate adjustment outside this function
    for i in range(12292):
        if longs[i] > 180:
            longs[i] = -(360 - longs[i])
        point = shapely.geometry.Point(longs[i], lats[i])
        circle = point.buffer(0.35*shrinkage_factor)
        patch = PolygonPatch(circle, facecolor=colors[i], edgecolor=colors[i], alpha=0.7, zorder=2)
        points.append(patch)
    #put the points in a collection and plot them
    collection = PatchCollection(points)
    ax.add_collection(collection)
    collection.set_color(colors)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, facecolor='green')
    plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), ax=ax)
    plt.show()


# def main():
#     long = load_csv_data('Longitd', "../scaled_scores_latlong_reproj_SJ.csv")
#     lat = load_csv_data('Latitud', "../scaled_scores_latlong_reproj_SJ.csv")
#     mat = load_matrix(12292, "../R_to_Py_connmat_reduced")
#     data = load_pc_data("../Patch_Con_Data_corrected_3_0-2000")
#
#     map_visualize(data[:, 1], lat, long, -85, -73, 19, 25)
#
#     # g = make_igraph_graph(mat)
#     # print(g.clusters(mode=igraph.STRONG))
#     # long = load_csv_data("Longitd", "../scaled_scores_latlong_reproj_SJ.csv")
#     # lat = load_csv_data("Latitud", "../scaled_scores_latlong_reproj_SJ.csv")
#     # graph_vis(g, 12292, 5000, True, long, lat)
#
#
# if __name__ == '__main__':
#     main()