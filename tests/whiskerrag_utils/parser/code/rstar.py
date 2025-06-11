from typing import List, Optional


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Rectangle:
    def __init__(self):
        self.min_x = float("inf")
        self.min_y = float("inf")
        self.max_x = float("-inf")
        self.max_y = float("-inf")

    def area(self) -> float:
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def overlaps(self, other: "Rectangle") -> bool:
        return not (
            self.max_x < other.min_x
            or self.min_x > other.max_x
            or self.max_y < other.min_y
            or self.min_y > other.max_y
        )


class RStarNode:
    MAX_ENTRIES = 4
    MIN_ENTRIES = 2

    def __init__(self, level: int = 0):
        self.points: List[Point] = []
        self.children: List[RStarNode] = []
        self.mbr: Rectangle = Rectangle()
        self.parent: Optional[RStarNode] = None
        self.level = level

    @property
    def is_leaf(self) -> bool:
        return self.level == 0

    def compute_mbr(self):
        self.mbr = Rectangle()

        if self.is_leaf:
            for point in self.points:
                self.mbr.min_x = min(self.mbr.min_x, point.x)
                self.mbr.min_y = min(selfmin_y, point.y)
                self.mbr.max_x = max(self.mbr.max_x, point.x)
                self.mbr.max_y = max(self.mbr.max_y, point.y)
        else:
            for child in self.children:
                self.mbr.min_x = min(self.mbr.min_x, child.mbr.min_x)
                self.mbr.min_y = min(self.mbr.min_y, child.mbr.min_y)
                self.mbr.max_x = max(self.mbr.max_x, child.mbr.max_x)
                self.mbr.max_y = max(self.mbr.max_y, child.mbr.max_y)


class RStarTree:
    def __init__(self):
        self.root = RStarNode()

    def insert(self, point: Point):
        leaf = self._choose_leaf(self.root, point)
        leaf.points.append(point)
        leaf.compute_mbr()

        if len(leaf.points) > RStarNode.MAX_ENTRIES:
            self._split_node(leaf)

        self._adjust_tree(leaf)

    def _choose_leaf(self, node: RStarNode, point: Point) -> RStarNode:
        if node.is_leaf:
            return node

        min_increase = float("inf")
        chosen = node.children[0]

        for child in node.children:
            increase = self._calculate_enlargement(child.mbr, point)
            if increase < min_increase:
                min_increase = increase
                chosen = child

        return self._choose_leaf(chosen, point)

    def _calculate_enlargement(self, mbr: Rectangle, point: Point) -> float:
        old_area = mbr.area()
        new_min_x = min(mbr.min_x, point.x)
        new_min_y = min(mbr.min_y, point.y)
        new_max_x = max(mbr.max_x, point.x)
        new_max_y = max(mbr.max_y, point.y)
        new_area = (new_max_x - new_min_x) * (new_max_y - new_min_y)
        return new_area - old_area

    def _split_node(self, node: RStarNode):
        new_node = RStarNode(node.level)

        if node.is_leaf:
            # Sort points by x-coordinate
            node.points.sort(key=lambda p: p.x)
            mid_index = len(node.points) // 2
            new_node.points = node.points[mid_index:]
            node.points = node.points[:mid_index]
        else:
            # Sort children by their MBR's center x-coordinate
            node.children.sort(key=lambda n: (n.mbr.min_x + n.mbr.max_x) / 2)
            mid_index = len(node.children) // 2
            new_node.children = node.children[mid_index:]
            node.children = node.children[:mid_index]

            for child in new_node.children:
                child.parent = new_node

        node.compute_mbr()
        new_node.compute_mbr()

        if node is self.root:
            new_root = RStarNode(node.level + 1)
            new_root.children.extend([node, new_node])
            node.parent = new_root
            new_node.parent = new_root
            self.root = new_root
            new_root.compute_mbr()
        else:
            new_node.parent = node.parent
            node.parent.children.append(new_node)
            node.parent.compute_mbr()

    def _adjust_tree(self, node: RStarNode):
        if node is self.root:
            return

        current = node
        while current is not self.root:
            current.parent.compute_mbr()
            if len(current.parent.children) > RStarNode.MAX_ENTRIES:
                self._split_node(current.parent)
            current = current.parent

    def search(self, query_mbr: Rectangle) -> List[Point]:
        return self._search_node(self.root, query_mbr)

    def _search_node(self, node: RStarNode, query_mbr: Rectangle) -> List[Point]:
        results = []

        if not node.mbr.overlaps(query_mbr):
            return results

        if node.is_leaf:
            for point in node.points:
                if (
                    query_mbr.min_x <= point.x <= query_mbr.max_x
                    and query_mbr.min_y <= point.y <= query_mbr.max_y
                ):
                    results.append(point)
        else:
            for child in node.children:
                results.extend(self._search_node(child, query_mbr))

        return results

    def delete(self, point: Point) -> bool:
        leaf = self._find_leaf(self.root, point)
        if not leaf:
            return False

        try:
            leaf.points.remove(point)
            leaf.compute_mbr()
            self._condense_tree(leaf)
            return True
        except ValueError:
            return False

    def _find_leaf(self, node: RStarNode, point: Point) -> Optional[RStarNode]:
        if node.is_leaf:
            for p in node.points:
                if p.x == point.x and p.y == point.y:
                    return node
            return None

        for child in node.children:
            if (
                child.mbr.min_x <= point.x <= child.mbr.max_x
                and child.mbr.min_y <= point.y <= child.mbr.max_y
            ):
                result = self._find_leaf(child, point)
                if result:
                    return result
        return None

    def _condense_tree(self, node: RStarNode):
        if node is self.root:
            if not node.is_leaf and len(node.children) == 1:
                self.root = node.children[0]
                self.root.parent = None
            return

        if len(node.points) < RStarNode.MIN_ENTRIES:
            parent = node.parent
            parent.children.remove(node)
            if node.is_leaf:
                for point in node.points:
                    self.insert(point)
            else:
                for child in node.children:
                    self.insert_node(child)
            node.parent.compute_mbr()
            self._condense_tree(parent)
        else:
            self._adjust_tree(node)

    def insert_node(self, node: RStarNode):
        for point in node.points:
            self.insert(point)
