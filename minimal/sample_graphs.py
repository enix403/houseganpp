from minimal.layout import LayoutGraphBuilder, LayoutGraph, NodeType

def one():
    bld = LayoutGraphBuilder()

    liv = bld.add_node(NodeType.LIVING_ROOM)

    r1 = bld.add_node(NodeType.BEDROOM)
    r2 = bld.add_node(NodeType.BEDROOM)
    kit = bld.add_node(NodeType.KITCHEN)
    bal = bld.add_node(NodeType.BALCONY)
    b1 = bld.add_node(NodeType.BATHROOM)

    bld.add_edge(liv, r1)
    bld.add_edge(liv, r2)
    bld.add_edge(liv, kit)
    bld.add_edge(liv, bal)
    bld.add_edge(liv, b1)

    g = bld.build()
    g.correct_doors()

    return g

def two():
    bld = LayoutGraphBuilder()

    liv = bld.add_node(NodeType.LIVING_ROOM)
    kit = bld.add_node(NodeType.KITCHEN)
    bal = bld.add_node(NodeType.BALCONY)
    r1 = bld.add_node(NodeType.BEDROOM)
    b1 = bld.add_node(NodeType.BATHROOM)
    r2 = bld.add_node(NodeType.BEDROOM)
    b2 = bld.add_node(NodeType.BATHROOM)
    r3 = bld.add_node(NodeType.BEDROOM)
    b3 = bld.add_node(NodeType.BATHROOM)

    bld.add_edge(liv, kit)
    bld.add_edge(liv, r1)
    bld.add_edge(liv, r2)
    bld.add_edge(liv, r3)

    bld.add_edge(r1, b1)
    bld.add_edge(r2, b2)
    bld.add_edge(r3, b3)
    bld.add_edge(r3, bal)

    g = bld.build()
    g.correct_doors()

    return g
