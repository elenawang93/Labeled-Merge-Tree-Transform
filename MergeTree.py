import bisect
import cv2
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sortedcontainers import *
from UnionFind import *
from DistanceUtils import *


def _ptProjectToGraph(pt, graph):
    '''
    project a point to its nearest edge on input graph

    pt: tuple[int, int]
    graph: nx.Graph
    '''
    minDist = float('inf')
    min_v0 = -1
    min_v1 = -1
    projected = [None, None]

    for v0, v1 in graph.edges():

        p0 = graph.nodes[v0]['pos']
        p1 = graph.nodes[v1]['pos']

        # if intersecting the line segment, then this is shortest path
        # else it will be the min of the endpoints
        intersection = intersectionToLine2Pts(pt, p0, p1)

        # where it intersects is on the line so only need to check domain
        maxX = max(p0[0], p1[0])
        minX = min(p0[0], p1[0])    

        # handle vertical edges
        maxY = max(p0[1], p1[1])
        minY = min(p0[1], p1[1])  

        if intersection[0] <= maxX and intersection[0] >= minX and intersection[1] <= maxY and intersection[1] >= minY:
            dist = distToLine2Pts(pt, p0, p1)
            matchType = 2
        else:
            dist_p0 = math.dist(pt, p0)
            dist_p1 = math.dist(pt, p1)

            if dist_p0 < dist_p1:
                matchType = 0
                dist = dist_p0
            else:
                matchType = 1
                dist = dist_p1
        
        # update if min dist is shortest
        if dist < minDist:
            minDist = dist
            min_v0, min_v1 = v0, v1
            
            # match/case if python 3.10+
            if matchType == 2:
                projected = intersection
            elif matchType == 0:
                projected = p0
            else:
                projected = p1
            
    return projected, min_v0, min_v1


def _projectNodesOnto(giver, receiver_orig, verbose=False):
    '''
    project all nodes of input giver graph onto receiver graph

    giver: nx.Graph
    receiver_orig: nx.Graph
    verbose: bool
    '''
    receiver = receiver_orig.copy()
    
    # NOTE this requires the nodes of the graph to be labelled from 0
    # project giver onto receiver (nodes should already be mapped)
    for pt, data in giver.nodes(data=True):

        # do not project if not extremal
        if not data['extremal']:
            continue
    
        projected, p_v0, p_v1 = _ptProjectToGraph(data['pos'], receiver)
        receiver.add_node(pt)
        receiver.nodes[pt]['pos'] = projected
        receiver.nodes[pt]['p_v0'] = p_v0
        receiver.nodes[pt]['p_v1'] = p_v1
        receiver.nodes[pt]['projected'] = True
        receiver.nodes[pt]['extremal'] = True

    return receiver


def detectExtremal(g_orig):
    '''
    detect extremal nodes in the graph

    g_orig: nx.Graph
    '''
    g = g_orig.copy()
    
    for n in g.nodes:
        hull = np.array([g.nodes[x]['pos'] for x in g.neighbors(n)]).astype(np.float32)
        if cv2.pointPolygonTest(hull, g.nodes[n]['pos'], False) < 0:
            # then this is outside of the hull
            g.nodes[n]['extremal'] = True
        else:
            g.nodes[n]['extremal'] = False
    
    return g


def prepareTwoGraphs(g1_orig, g2_orig, verbose=False):
    '''
    project only extremal nodes the graph onto each other
    requires integer node names to allow for proper offset of node names
    end by flattening points into the edge (build edges to the endpoints)

    g1_orig: nx.Graph
    g2_orig: nx.Graph
    verbose: bool
    '''
    g1 = g1_orig.copy()
    g2 = nx.relabel_nodes(g2_orig, lambda x: x + max(g1.nodes()) + 1)

    g1 = detectExtremal(g1)
    g2 = detectExtremal(g2)

    g2_proj = _projectNodesOnto(g1, g2, verbose=verbose)
    g1_proj = _projectNodesOnto(g2, g1, verbose=verbose)

    # flatten point into the edge
    g1_proj = _flattenToEndpoints(g1_proj, verbose=verbose)
    g2_proj = _flattenToEndpoints(g2_proj, verbose=verbose)
    
    return g1_proj, g2_proj


def prepareThreeGraphs(g1_orig, g2_orig, g3_orig, verbose=False):
    '''
    extend to project a triple of graphs onto each other, using extremal points only

    g1_orig: nx.Graph
    g2_orig: nx.Graph
    g3_orig: nx.Graph
    verbose: bool
    '''
    g1 = g1_orig.copy()
    g2 = nx.relabel_nodes(g2_orig, lambda x: x + max(g1.nodes()) + 1)
    g3 = nx.relabel_nodes(g3_orig, lambda x: x + max(g2.nodes()) + 1)

    g1 = detectExtremal(g1)
    g2 = detectExtremal(g2)
    g3 = detectExtremal(g3)

    # project onto, since ptProject only works on edges, floating points dont hurt it until flatten
    g1_proj = _projectNodesOnto(g2, g1, verbose=verbose)
    g1_proj = _projectNodesOnto(g3, g1_proj, verbose=verbose)
    g2_proj = _projectNodesOnto(g1, g2, verbose=verbose)
    g2_proj = _projectNodesOnto(g3, g2_proj, verbose=verbose)
    g3_proj = _projectNodesOnto(g1, g3, verbose=verbose)
    g3_proj = _projectNodesOnto(g2, g3_proj, verbose=verbose)

    # flatten point into the edge
    g1_proj = _flattenToEndpoints(g1_proj, verbose=verbose)
    g2_proj = _flattenToEndpoints(g2_proj, verbose=verbose)
    g3_proj = _flattenToEndpoints(g3_proj, verbose=verbose)

    return g1_proj, g2_proj, g3_proj


def _computeNodeHeights(graph, filtration, precision=5):
    '''
    given a filtration line and direction, compute heights of each node and return as dict of tuple (height, projected)

    graph: nx.Graph
    filtration: tuple[tuple[float, float], tuple[float, float], int]
    precision: int
    '''
    # defining a line as 2 points and an inversion flag
    p0 = filtration[0]
    p1 = filtration[1]
    angle_sign = filtration[2] # based on the critical angle
    
    # only need to iterate in sorted order (so no need for SortedDicts/BSTs)
    # could also save heights into the graph properties, but for now utilizing this other data structure
    heights = {}
    for node, data in graph.nodes(data=True):
        # need to calculate when the existing point is "above" or "below"
        # so it's not just a raw absolute distance to line, but tracking position using
        # y < f(x) or y > f(x)
        height = round(angle_sign * signedDistToLine2Pts(data['pos'], p0, p1), precision)
        projected = data.get('projected', False)
        
        # false comes first in sort order, so extremal comes first
        nonExtremal = not data.get('extremal', False)
        heights[node] = (height, nonExtremal, projected)
    
    return heights


def getSortedNodeHeights(graph, filtration, precision=5):
    '''
    compute heights of each node given filtration line and return as sorted list of node height tuples, rounded to given precision

    graph: nx.Graph
    filtration: tuple[tuple[float, float], tuple[float, float], int]
    precision: int
    '''
    heightTuples = _computeNodeHeights(graph, filtration, precision=precision)
    return [(x[0], x[1][0]) for x in sorted(heightTuples.items(), key=lambda x:x[1])]


def buildMergeTreeAllPoints(g_orig, filtration, infAdjust=1, precision=5, show=True, size=0, verbose=False):
    '''
    build merge tree of all points in the graph

    g_orig: nx.Graph
    filtration: tuple[tuple[float, float], tuple[float, float], int]
    infAdjust: int
    precision: int
    show: bool
    size: int
    verbose: bool
    '''
    # make copy of graph since we save node properties into graph
    
    g = g_orig.copy()
    
    # digraph for LCA calcs and it's a tree
    mt = nx.DiGraph()
    
    # to handle special indexes post projections (has nodes named >n)
    if size is not None:
        uf = UnionFind(size, verbose=verbose)
    else:
        uf = UnionFind(g.number_of_nodes(), verbose=verbose)
        
    visited = set()
    numComponents = 0
    heights = getSortedNodeHeights(g, filtration, precision)

    # this is the first node of min height since list
    topMerge = heights[0][0]
    for node, height in heights:
        # track visited nodes (helps deal with equal heights)
        if verbose:
            print(f"now processing node{node}, with {numComponents} already found components")
        visited.add(node)

        # check to see if the node has been told it is the endpoint of a previous grouping (the endpoint of an already found edge)
        # perform find to make sure these groupings are not the same
        possibleGroups = g.nodes[node].get('groups', [])

        # if this edge has never been told anything, no existing edges
        # add this node in merge tree as start of a new branch
        if possibleGroups == []:
            if verbose:
                print(f"{node} is unconnected, about to add {numComponents}, {height}")
            mt.add_node(node, pos=(numComponents, height), height=height)
            numComponents += 1

        else:
            # iterate through possible groups via unionFind to determine if this is a merge point or one connected component
            componentSet = set()
            for possibleGroup in possibleGroups:
                componentSet.add(uf.find(possibleGroup))

            componentList = list(componentSet)
            
            if verbose:
                print( f"received {componentList} membership")
            
            # if they are all the same group, this node is also part of this group
            # put every point on merge tree since this is the full method, even if already connected
            if len(componentList) == 1:
                myRoot = componentList.pop()
                uf.union(node, myRoot)

                if verbose:
                    print(f"although connected, key label{node}, existing connected to {myRoot}, adding still")
                mt.add_node(node, pos=(mt.nodes[myRoot]['pos'][0], height), height=height)
                mt.add_edge(node, myRoot)

                # change the root to represent the current head of merge point
                if verbose:
                    print(f"rerooting component {myRoot} to {node}")
                uf.rerootComponent(node, node)

                # this point could be a top merge point to infinity root
                topMerge = node
            else:
                # else this node is the merge point, add on merge tree and perform union
                if verbose:
                    print(f"about to add {numComponents}, {height}, updating topMerge")
                
                topMerge = node
                
                mt.add_node(node, pos=(numComponents-len(componentList), height), height=height)
                
                for component in componentList:
                    componentRoot = uf.find(component)
                    if verbose:
                        print(f"unioning node{node} and componentRoot of node {componentRoot}")
                        
                    # union each component
                    uf.union(node, componentRoot)
                    # track on merge tree
                    mt.add_edge(node, componentRoot)

                    if verbose:
                        print(f"rerooting {componentRoot} to merge point {node}")
                        
                    # change the root to represent the current head of merge point
                    uf.rerootComponent(node, node)
                    
                    numComponents -= 1
                    
        # pass along the finalized group to all the edges above
        myGroup = uf.find(node)
        for neighbor in nx.all_neighbors(g, node):
            # lower height neighbors seen before
            if verbose:
                print( f"neighbor{neighbor}")
            if neighbor in visited:
                if verbose:
                    print(f"visited{neighbor} already")
                continue

            # pass new info
            groups = g.nodes[neighbor].get('groups', [])
            g.nodes[neighbor]['groups'] = groups + [myGroup]
        
    # add final "inf" point, but visualize as height max(height+1?)
    infHeight = heights[len(heights)-1][1] + infAdjust
    mt.add_node('inf', pos=(0, infHeight), height=float('inf'))
    mt.add_edge('inf', topMerge)
    
    if show:
        nx.draw(mt, pos=nx.get_node_attributes(mt, 'pos'), node_color='#FFFFFF', with_labels=True)
        plt.show()
    
    return mt


def _cacheEdgeOfProjectedNodes(graph):
    '''
    utility for sortedFlatten but not used by merge tree construction

    graph: nx.Graph
    '''
    cache = {}
    for node in graph.nodes():
        if not 'p_v0' in graph.nodes[node]:
            continue
        
        p_v0 = graph.nodes[node]['p_v0']    
        p_v1 = graph.nodes[node]['p_v1']
        
        edgeKey = tuple(sorted((p_v0, p_v1)))
        
        if edgeKey in cache:
            cache[edgeKey].append(node)
        else:
            cache[edgeKey] = [node]
        
    return cache


def _sortedFlatten(graph_orig, verbose=False):
    '''
    complete flattening, A-(C,D)-B to A-C-D-B, but not used by merge tree construction

    graph_orig: nx.Graph
    verbose: bool
    '''
    graph = graph_orig.copy()
    cache = _cacheEdgeOfProjectedNodes(graph)
    
    for edge in cache.keys():        
        # tuple is already sorted
        
        if verbose:
            print( f"flattening edge {edge}")
        root = edge[0]
        rootPos = graph.nodes[root]['pos']
        
        # sort by distance
        dist = {}
        for node in cache[edge]:
            dist[node] = math.dist(graph.nodes[node]['pos'], rootPos)
        
        dist = dict(sorted(dist.items(), key=lambda x: x[1]))
        sortedNodes = dist.keys()
        
        # remove original edge
        graph.remove_edge( edge[0], edge[1])
        prevNode = edge[0]
        
        # add each line segment
        for node in sortedNodes:
            if verbose:
                print( f"adding from {prevNode} to {node}")
            graph.add_edge( prevNode, node )
            prevNode = node
    
        # add final segment
        if verbose:
            print( f"adding from {prevNode} to endpoint {edge[1]}")
        graph.add_edge( prevNode, edge[1])
    
    return graph


def _flattenToEndpoints(graph_orig, verbose=False):
    '''
    flatten with many edges with the same endpoints, A-(C,D)-B becomes A-B, A-C-B, A-D-B, currently used in merge tree construction

    graph_orig: nx.Graph
    verbose: bool
    '''
    graph = graph_orig.copy()

    for node, data in graph.nodes(data=True):
        if data.get('projected', False):
            
            if verbose:
                print(f"handling node {node}, adding edges to {data['p_v0']} and to {data['p_v0']}")
    
            # add the two edges
            graph.add_edge( node, data['p_v0'] )
            graph.add_edge( node, data['p_v1'] )

    return graph


def _mtToOtherGraph(mt_orig, graph_orig, verbose=False):
    '''
    take nodes of mt and only keep projections of those nodes on the graph

    mt_orig: nx.DiGraph
    graph_orig: nx.Graph
    verbose: bool
    '''
    graph = graph_orig.copy()
    
    # remove unneeded projected points
    mt_nodes = mt_orig.nodes()
    
    # using graph_orig to remove nodes on graph (the return value copy)
    for node in graph_orig.nodes():
        if node not in mt_nodes and graph_orig.nodes[node].get('projected', False):
            if verbose:
                print(f"removing projected but unimportant node {node}")
            graph.remove_node(node)
        
    # full flattening of the remainder key nodes    
    graph = _flattenToEndpoints(graph, verbose=verbose)
        
    return graph


def computeCriticalAngles(graph):
    '''
    get sorted set of all critical angles for a graph

    graph: nx.Graph
    '''
    angles = SortedSet()
    for v0, v1 in graph.edges():
        p0 = graph.nodes[v0]['pos']
        p1 = graph.nodes[v1]['pos']
        
        # don't add angle if projection is self
        if np.isclose(p0, p1, atol=1e-6).all():
            continue
        
        y = p1[1]-p0[1]
        x = p1[0]-p0[0]
        
        # add angle + its supplement
        if y == 0:
            # horizontal line, normal is vertical
            angles.add(math.pi/2)
            angles.add(3*math.pi/2)
            
        else:
            # this is the angle formed to the normal, the domain adjusted to 0<theta<2*pi
            atan = math.atan(-x/y) % (2*math.pi)
            angles.add(atan)
            angles.add((math.pi + atan) % (2*math.pi))

    return angles


def computeAllCriticalAngles(g1, g2):
    '''
    union all critical angles of two graphs in sorted set

    g1: nx.Graph
    g2: nx.Graph
    '''
    return computeCriticalAngles(g1).union(computeCriticalAngles(g2))


def computeAllAngles(g1, g2):
    '''
    get all critical angles of 2 graphs, sorted, as well as midpoints between those critical angles

    g1: nx.Graph
    g2: nx.Graph
    '''
    critical = list(computeAllCriticalAngles(g1, g2))

    # handling boundary case
    critical.append(critical[0] + 2*math.pi)

    window = sliding_window_view(critical, window_shape=2)
    midpoints = [(x[0]+x[1])/2 % (2*math.pi) for x in window]

    # clean up the boundary case
    critical = critical[:-1]

    # critical angles already sorted, this final sort is for the boundary condition
    return critical, sorted(midpoints)


def findFiltration(theta, origin=(0,0)):
    '''
    get line of theta slope going through origin point, also track an inversion flag for dist

    theta: float
    origin: tuple[float, float]
    '''
    # could assert this domain but can just adjust for it
    if theta >= 2*math.pi:
        theta = theta % (2*math.pi)
    
    if theta == 0:
        p1 = (origin[0], origin[1]+1)
    else:    
        # slope is normal to the angle
        slope = -1 / math.tan(theta)
        p1 = (origin[0]+1, origin[1]+slope)
    
    # flip distance function for certain angles
    # here, since "left" in our signed dist function is positive, inverting here
    sign = 1 if theta <= math.pi and theta > 0 else -1
    
    return (origin, p1, sign)


def getHeightMatrix(mt, verbose=False):
    '''
    LCA from networkx, assuming it to be better than Eulerian tour + spare table min range query, returning a matrix of pairwise LCA heights and then the matrix of LCA nodes

    mt: nx.DiGraph
    verbose: bool
    '''
    # remove inf node
    size = mt.number_of_nodes()-1
    heightMatrix = np.zeros((size,size))
    nodeMatrix = np.zeros((size,size))
    
    # map actual node indexes to sorted matrix indexes
    nodesList=list(mt.nodes)
    nodesList.remove('inf')
    allNodes = sorted(nodesList)
    nodesDict = dict( zip(allNodes, range(len(allNodes))) )
    
    for LCA in nx.tree_all_pairs_lowest_common_ancestor(mt):
        # this is of the form ( (node1, node2), LCA )
        # handle root separately
        if LCA[0][0]=='inf' or LCA[0][1]=='inf':
            continue
        
        i = nodesDict[LCA[0][0]]
        j = nodesDict[LCA[0][1]]
        heightMatrix[i,j] = mt.nodes[LCA[1]]['height']
        heightMatrix[j,i] = mt.nodes[LCA[1]]['height']
        
        nodeMatrix[i,j] = LCA[1]
        nodeMatrix[j,i] = LCA[1]
        
    heightMatrix = np.matrix(heightMatrix)
    nodeMatrix = np.matrix(nodeMatrix)
    
    if verbose:
        print(nodesDict)
        print(heightMatrix)
        print(nodeMatrix)
        
    return heightMatrix, nodeMatrix


def _deleteRowColumn(arr, deleteIdxList):
    '''
    delete row and col x for each idx in delete list

    arr: np.array
    deleteIdxList: list[int]
    '''
    # delete row and col x for each idx in delete list
    arr = np.delete(arr, deleteIdxList, 0)
    arr = np.delete(arr, deleteIdxList, 1)
    return arr


def calcDistanceMatrix(mt1, mt2, g1, g2, extremalOnly=True, verbose=False):
    '''
    calculate difference matrix of LCA heights, also return LCA nodes of mt1 and mt2

    mt1: nx.DiGraph
    mt2: nx.DiGraph
    g1: nx.Graph
    g2: nx.Graph
    extremalOnly: bool
    verbose: bool
    '''
    h1, nodes1 = getHeightMatrix(mt1, verbose=verbose)
    h2, nodes2 = getHeightMatrix(mt2, verbose=verbose)
    
    finalNodes = set(g1.nodes) | set(g2.nodes)

    # zero out any pairs with a non-extremal point
    if extremalOnly:
        # get all non extremal nodes, get their projected IDs and matrix space
        nonExtremalg1 = [x[0] for x in nx.get_node_attributes(g1, 'extremal').items() if not x[1]]
        nonExtremalg2 = [x[0] for x in nx.get_node_attributes(g2, 'extremal').items() if not x[1]]

        # map actual node names to sorted matrix indexes
        # each matrix has all their points + the projected extremal
        m1NodesList = list(mt1.nodes)
        m1NodesList.remove('inf')
        m1Nodes = sorted(m1NodesList)

        m2NodesList = list(mt2.nodes)
        m2NodesList.remove('inf')
        m2Nodes = sorted(m2NodesList)
                
        # dict from node to index based on sort
        m1Dict = dict(zip(m1Nodes, range(len(m1Nodes))))
        m2Dict = dict(zip(m2Nodes, range(len(m2Nodes))))
        
        # map nodes to idx to delete
        g1DeleteIdx = [m1Dict[x] for x in nonExtremalg1]
        g2DeleteIdx = [m2Dict[x] for x in nonExtremalg2]
        
        h1 = _deleteRowColumn(h1, g1DeleteIdx)
        h2 = _deleteRowColumn(h2, g2DeleteIdx)
        
        nodes1 = _deleteRowColumn(nodes1, g1DeleteIdx)
        nodes2 = _deleteRowColumn(nodes2, g2DeleteIdx)
        
        # remove nonextremal from finalNodes
        finalNodes -= set(nonExtremalg1)
        finalNodes -= set(nonExtremalg2)
        
    assert h1.shape==h2.shape, "height matrix shapes not equal"
    assert nodes1.shape==nodes2.shape, "LCA matrix shapes not equal"
    
    diff = np.subtract(h2, h1)
    if verbose:
        print(diff)
            
    return diff, nodes1, nodes2, sorted(finalNodes)


def getUnitVector(angle):
    '''
    get 2D unit vector of input angle

    angle: float
    '''
    x = math.cos(angle)
    y = math.sin(angle)
    return (x,y)


def getMidpointKey(arr, target):
    '''
    get the midpoint key of the region that contains the target

    arr: list[float]
    target: float
    '''
    b = bisect.bisect_left( arr, target )
    if b>=len(arr) or b == 0:
        return ((arr[-1]+arr[0])/2 + math.pi) % (math.pi*2)
    else:
        return (arr[b]+arr[b-1])/2 


def _innerProduct(unitVec, pos):
    '''
    compute inner product of 2D vector with position

    unitVec: tuple[float, float]
    pos: tuple[float, float]
    '''
    return(unitVec[0] * pos[0] + unitVec[1] * pos[1])


def computeGraphDistanceAtAngle(angle, G1, G2, criticalDict, midpointDict, verbose=False):
    '''
    compute distance of two embedded graphs at a given angle using cached precomputation and inferring all other angles

    angle: float
    G1: nx.Graph
    G2: nx.Graph
    criticalDict: dict[float, np.matrix]
    midpointDict: dict
    verbose: bool
    '''
    angle = angle % (math.pi*2)
    
    criticalAngles = list(criticalDict.keys())
    if angle in criticalAngles:
        if verbose:
            print(f'using critical angle cache for: {angle}\n')
        
        diff = criticalDict[angle]['diff']
        infnorm = np.max(np.abs(diff))
        data = criticalDict[angle]['data']
    else:
        # convert angle to unit vector
        unitVec = getUnitVector(angle)
        
        # find the computed midpoint matrix to work off of
        key = getMidpointKey(criticalAngles, angle)
        
        # precision rounding
        key = math.floor(key*1e9) / 1e9 + 0.01
        
        if verbose:
            print(f'found midpoint key {key} for angle {angle}\n')
                
        # intentionally ignoring the heights computed for the midpoint
        LCA0 = midpointDict[key]['LCA0']
        LCA1 = midpointDict[key]['LCA1']
        
        # both matrices should be same shape
        assert(LCA0.shape==LCA1.shape)
        A0 = np.matrix(np.zeros(LCA0.shape))
        A1 = np.matrix(np.zeros(LCA1.shape))
        
        # inner product of unit vector and LCA position
        for i in range(A0.shape[0]):
            for j in range(A0.shape[1]):
                pos0 = G1.nodes[LCA0[i,j]]['pos']
                A0[i,j] = _innerProduct(unitVec, pos0)
                
                pos1 = G2.nodes[LCA1[i,j]]['pos']
                A1[i,j] = _innerProduct(unitVec, pos1)
        
        diff = np.subtract(A1, A0)
        infnorm = np.max(np.abs(diff))
        data = midpointDict[key]['data']
    
    return infnorm, diff, data


def computeDistanceAtAngleFromMT(g1_orig, g2_orig, angle, precision=5, show=False, verbose=False):
    '''
    given an angle and two graphs, compute full difference matrix and LCA node matrix via extremal technique

    g1_orig: nx.Graph
    g2_orig: nx.Graph
    angle: float
    precision: int
    show: bool
    verbose: bool
    '''
    g1 = g1_orig.copy()
    g2 = g2_orig.copy()
    
    filtration = findFiltration(angle)
    
    mt1 = buildMergeTreeAllPoints(g1, filtration, precision=precision, show=show, size=max(g1.nodes())+1, verbose=verbose)
    mt2 = buildMergeTreeAllPoints(g2, filtration, precision=precision, show=show, size=max(g2.nodes())+1, verbose=verbose)
        
    return calcDistanceMatrix(mt1, mt2, g1, g2, extremalOnly=True, verbose=verbose)


def computeDistanceFull(g1, g2, precision=5, show=False, verbose=False):
    '''
    calculate all necessary pre-computations of height and LCA matrices for every critical angle and midpoint of two embedded graphs

    g1: nx.Graph
    g2: nx.Graph
    precision: int
    show: bool
    verbose: bool
    '''
    criticalAngles, midpoints = computeAllAngles(g1, g2)
    
    c_dict = {}
    for angle in criticalAngles:
        if show:
            print(f"using critical angle: {angle}")
        diff, LCA0, LCA1, data = computeDistanceAtAngleFromMT(g1, g2, angle, precision=precision, show=show, verbose=verbose)
        c_dict[angle] = {'diff':diff, 'LCA0':LCA0, 'LCA1':LCA1, 'data':data}
    
    m_dict = {}
    for angle in midpoints:
        angle = math.floor(angle * 1e9) / 1e9 + 0.01
        if show:
            print(f"using midpoint: {angle}")
        diff, LCA0, LCA1, data = computeDistanceAtAngleFromMT(g1, g2, angle, precision=precision, show=show, verbose=verbose)
        m_dict[angle] = {'diff':diff, 'LCA0':LCA0, 'LCA1':LCA1, 'data':data}
    
    return c_dict, m_dict


def getMaxIndex(diff_matrix):
    '''
    get the indexes of the maximum values in the difference matrix

    diff_matrix: np.matrix
    '''
    abs_diff_matrix = np.abs(diff_matrix)
    max_diff = np.max(abs_diff_matrix)
    max_indexes = np.argwhere(abs_diff_matrix == max_diff)
    max_indexes_list = [tuple(index) for index in max_indexes]
    return max_indexes_list


def get3GraphDistance(G1_orig, G2_orig, G3_orig, precision=5, plot=True, show=False, verbose=False, xMin=0, xMax=2*math.pi, n=10000):
    '''
    get the 3 pairwise distances between each pair of 3 embedded graphs over linspace of n points from xMin to xMax, full driver code, plot to show graph, show is for internal merge trees

    G1_orig: nx.Graph
    G2_orig: nx.Graph
    G3_orig: nx.Graph
    precision: int
    plot: bool
    show: bool
    verbose: bool
    xMin: float
    xMax: float
    n: int
    '''
    G1, G2, G3 = prepareThreeGraphs(G1_orig, G2_orig, G3_orig, verbose=verbose)

    # precache given the prepared graphs
    c_dict1, m_dict1 = computeDistanceFull(G1, G2, precision=precision, show=show, verbose=verbose)
    c_dict2, m_dict2 = computeDistanceFull(G1, G3, precision=precision, show=show, verbose=verbose)
    c_dict3, m_dict3 = computeDistanceFull(G2, G3, precision=precision, show=show, verbose=verbose)

    X = np.linspace(xMin, xMax, n)
    Y1, Y1diffs, Y1data = zip(*[computeGraphDistanceAtAngle(x, G1, G2, c_dict1, m_dict1) for x in X])
    Y2, Y2diffs, Y2data = zip(*[computeGraphDistanceAtAngle(x, G1, G3, c_dict2, m_dict2) for x in X])
    Y3, Y3diffs, Y3data = zip(*[computeGraphDistanceAtAngle(x, G2, G3, c_dict3, m_dict3) for x in X])

    # temp maps
    Y1maps = [{i:x for i, x in enumerate(maps)} for maps in Y1data]
    Y2maps = [{i:x for i, x in enumerate(maps)} for maps in Y2data]
    Y3maps = [{i:x for i, x in enumerate(maps)} for maps in Y3data]
    
    # use the dicts to map out extremal
    Y1data = [[(Y1map[i], Y1map[j]) for i, j in getMaxIndex(diffM) if Y1map[i] <= Y1map[j]] for diffM, Y1map in zip(Y1diffs, Y1maps)]
    Y2data = [[(Y2map[i], Y2map[j]) for i, j in getMaxIndex(diffM) if Y2map[i] <= Y2map[j]] for diffM, Y2map in zip(Y2diffs, Y2maps)]
    Y3data = [[(Y3map[i], Y3map[j]) for i, j in getMaxIndex(diffM) if Y3map[i] <= Y3map[j]] for diffM, Y3map in zip(Y3diffs, Y3maps)]
    
    if plot:
        plt.scatter(X, Y1, marker=',', s=1)
        plt.scatter(X, Y2, marker=',', s=1)
        plt.scatter(X, Y3, marker=',', s=1)
        plt.xlim([xMin, xMax])
        tick_positions = np.arange(xMin, xMax + np.pi/2, np.pi/2)
        # format in radians
        tick_labels = []
        for i in range(int(xMin//(np.pi/2)), int(xMax//(np.pi/2) + 1)):
            if i % 2 == 0:
                if i == 2:
                    tick_labels.append(r'$\pi$')
                else: 
                    tick_labels.append(rf'{i//2}$\pi$')
            elif i == 1:
                tick_labels.append(f'π/2')
            else:
                tick_labels.append(rf'{i}$\pi$/2' if i != 0 else '0')
        plt.xticks(tick_positions, tick_labels)
        plt.xlabel('Height Direction')
        plt.show()
    
    return X, Y1, Y2, Y3, Y1data, Y2data, Y3data, m_dict1, m_dict2, m_dict3


def get2GraphDistance(G1_orig, G2_orig, precision=5, plot=True, show=False, verbose=False, xMin=0, xMax=2*math.pi, n=10000):
    '''
    get pairwise distances of two embedded graphs over linspace of n points from xMin to xMax, full driver code, plot to show graph, show is for internal merge trees

    G1_orig: nx.Graph
    G2_orig: nx.Graph
    precision: int
    plot: bool
    show: bool
    verbose: bool
    xMin: float
    xMax: float
    n: int
    '''
    G1, G2,  = prepareTwoGraphs(G1_orig, G2_orig, verbose=verbose)

    # precache given the prepared graphs
    c_dict, m_dict = computeDistanceFull(G1, G2, precision=precision, show=show, verbose=verbose)

    X = np.linspace(xMin, xMax, n)
    Y, Ydiffs, Ydata = zip(*[computeGraphDistanceAtAngle(x, G1, G2, c_dict, m_dict) for x in X])

    # temp maps
    Ymaps = [{i:x for i, x in enumerate(maps)} for maps in Ydata]
    
    # use the dicts to map out extremal
    Ydata = [[(Ymap[i], Ymap[j]) for i, j in getMaxIndex(diffM) if Ymap[i] <= Ymap[j]] for diffM, Ymap in zip(Ydiffs, Ymaps)]

    if plot:
        plt.scatter(X, Y, marker=',', s=1)
        plt.xlim([xMin, xMax])
        tick_positions = np.arange(xMin, xMax + np.pi/2, np.pi/2)
        # format in radians
        tick_labels = []
        for i in range(int(xMin//(np.pi/2)), int(xMax//(np.pi/2) + 1)):
            if i % 2 == 0:
                if i == 2:
                    tick_labels.append(r'$\pi$')
                else: 
                    tick_labels.append(rf'{i//2}$\pi$')
            elif i == 1:
                tick_labels.append(rf'$\pi$/2')
            else:
                tick_labels.append(rf'{i}$\pi$/2' if i != 0 else '0')
        plt.xticks(tick_positions, tick_labels)
        plt.xlabel('Height Direction')
        plt.show()

    return X, Y, Ydata, m_dict
