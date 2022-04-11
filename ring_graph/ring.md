# Synthetic ring graph

The function create_ring creates a ring is with all edge directed into the same direction. A subgraph is attached on the ring every p nodes. In total, every
subgraph is repeated n times. The nodes of the ring are labeled r1 to rp having the following properties:

    weight = 1
    attr1 = 0.9
    attr2 = 0.6

The following subgraphs are attached to the ring

- bell
- tree
- star

The subgrahps are explained in more detail below.

## bell
a directed semi-complete graph with n nodes (b1) having incoming egdes from the other n nodes (b2). And having both incoming and outgoing
edge between nodes in the same group (b1, b2) with the following properties:

    edges weight between b1 and b2: 1
    edge weight between b1: 0.8
    edge_weight between b2: 0.6

    node label: b1 for nodes with incoming edges
                b2 for nodes with outgoing edges
    note attributes:
        b1: attr1 = 0.3, attr2 = 0.7
        b2: attr1 = 0.5, attr2 = 0.3

The first b1 node is relabel to b3 and used to attach to the ring

## star

creates a star of size n having one central node (s1) that is attached to two node types (roles) s2 and s3 resulting in a star shape subgraph. The edge are  from s1 to s2 and from s3 to s1. The nodes and edges have the below properties. 

    s1 - attr1: 0.5     attr2: 0.75
    s2 - attr1: 0.3     attr2: 0.4
    s3 - attr1: 0.7     attr2: 0.6

    weight s1 - s2 = 0.9
    weight s1 - s3 = 0.5

## tree
Creates a directed rooted tree. Every node has s children, where the first s/2 children (si) have an incoming edge from the parent and the second s/2 children (so) have an outgoing edges to the parent. The tree depth is equal to the specified depth. 
The graph has the following properties:

    rootnode:   attr1 = 0.2, attr2 = 0.3

    si:         attr1 = 0.4, attr2 = 0.9
    so:         attr1 = 0.1, attr2 = 0.6

    weight of si = 0.7,
    weight of so = 0.5


