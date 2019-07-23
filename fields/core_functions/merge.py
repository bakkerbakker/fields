# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:37:25 2019

@author: jesse bakker
"""

### Segmentation merge functions
from skimage.future import graph
import numpy as np


### Segmentation merge functions, using a Region Adjacency Graph (this merge step needs refinement)
### code from: http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rag_merge.html#sphx-glr-auto-examples-segmentation-plot-rag-merge-py
# gets called within the merge_RAGheirarchical function
def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

# gets called within the merge_RAGheirarchical function
def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])

    
# 'merge_RAGheirarchical_threshold': 0.5

def merge_RAG_hierarchical(input_img, input_labels, threshold):

    # create Region Adjacency Graph with rag mean color function
    g = graph.rag_mean_color(input_img, input_labels)

    labels2 = graph.merge_hierarchical(input_labels, g, thresh=threshold, 
                                       rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)
    print("labels2 complete")
    # This print statement will print out the array of merged segments
    # print(labels2[0])

#     out = color.label2rgb(labels2, input_img, kind='avg')
#     out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
#     fig = plt.figure(figsize=(12,12))
#     plt.imshow(out)
#     plt.show()
    
    return labels2

#### Combined function for workflow
def merge_segments_func(input_img, input_labels, **merge_inputs):
    # set input parameters
    RAG_hierarchical_threshold = merge_inputs['merge_hierarchical_threshold']
    # See merge.py for details on the merge/weight functions
    merged_labels = merge_RAG_hierarchical(input_img = input_img,
                                           input_labels = input_labels, 
                                           threshold=RAG_hierarchical_threshold)
    
    return merged_labels