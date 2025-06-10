import java.util.*;

public class rstar {
    private static final int MAX_ENTRIES = 4;
    private static final int MIN_ENTRIES = 2;

    static class Point {
        double x, y;

        Point(double x, double y) {
            this.x = x;
            this.y = y;
        }
    }

    static class Node {
        List<Point> points;
        List<Node> children;
        Node parent;
        int level;
        Rectangle mbr;

        Node(int level) {
            this.points = new ArrayList<>();
            this.children = new ArrayList<>();
            this.level = level;
            this.mbr = new Rectangle();
            this.parent = null;
        }

        boolean isLeaf() {
            return level == 0;
        }

        void computeMBR() {
            mbr.minX = Double.POSITIVE_INFINITY;
            mbr.minY = Double.POSITIVE_INFINITY;
            mbr.maxX = Double.NEGATIVE_INFINITY;
            mbr.maxY = Double.NEGATIVE_INFINITY;

            if (isLeaf()) {
                for (Point p : points) {
                    mbr.minX = Math.min(mbr.minX, p.x);
                    mbr.minY = Math.min(mbr.minY, p.y);
                    mbr.maxX = Math.max(mbr.maxX, p.x);
                    mbr.maxY = Math.max(mbr.maxY, p.y);
                }
            } else {
                for (Node child : children) {
                    mbr.minX = Math.min(mbr.minX, child.mbr.minX);
                    mbr.minY = Math.min(mbr.minY, child.mbr.minY);
                    mbr.maxX = Math.max(mbr.maxX, child.mbr.maxX);
                    mbr.maxY = Math.max(mbr.maxY, child.mbr.maxY);
                }
            }
        }
    }

    static class Rectangle {
        double minX, minY, maxX, maxY;

        Rectangle() {
            this.minX = Double.POSITIVE_INFINITY;
            this.minY = Double.POSITIVE_INFINITY;
            this.maxX = Double.NEGATIVE_INFINITY;
            this.maxY = Double.NEGATIVE_INFINITY;
        }

        double area() {
            return (maxX - minX) * (maxY - minY);
        }

        boolean overlaps(Rectangle other) {
            return !(maxX < other.minX || minX > other.maxX ||
                    maxY < other.minY || minY > other.maxY);
        }
    }

    private Node root;

    public rstar() {
        root = new Node(0);
    }

    public void insert(Point point) {
        Node leaf = chooseLeaf(root, point);
        leaf.points.add(point);
        leaf.computeMBR();

        if (leaf.points.size() > MAX_ENTRIES) {
            splitNode(leaf);
        }

        adjustTree(leaf);
    }

    private Node chooseLeaf(Node node, Point point) {
        if (node.isLeaf())
            return node;

        double minIncrease = Double.POSITIVE_INFINITY;
        Node chosen = null;

        for (Node child : node.children) {
            double increase = calculateEnlargement(child.mbr, point);
            if (increase < minIncrease) {
                minIncrease = increase;
                chosen = child;
            }
        }

        return chooseLeaf(chosen, point);
    }

    private double calculateEnlargement(Rectangle mbr, Point point) {
        double oldArea = mbr.area();
        double newMinX = Math.min(mbr.minX, point.x);
        double newMinY = Math.min(mbr.minY, point.y);
        double newMaxX = Math.max(mbr.maxX, point.x);
        double newMaxY = Math.max(mbr.maxY, point.y);
        double newArea = (newMaxX - newMinX) * (newMaxY - newMinY);
        return newArea - oldArea;
    }

    private void splitNode(Node node) {
        Node newNode = new Node(node.level);

        if (node.isLeaf()) {
            node.points.sort((a, b) -> Double.compare(a.x, b.x));
            int midIndex = node.points.size() / 2;
            newNode.points = new ArrayList<>(node.points.subList(midIndex, node.points.size()));
            node.points = new ArrayList<>(node.points.subList(0, midIndex));
        } else {
            node.children.sort((a, b) -> {
                double centerA = (a.mbr.minX + a.mbr.maxX) / 2;
                double centerB = (b.mbr.minX + b.mbr.maxX) / 2;
                return Double.compare(centerA, centerB);
            });

            int midIndex = node.children.size() / 2;
            newNode.children = new ArrayList<>(node.children.subList(midIndex, node.children.size()));
            node.children = new ArrayList<>(node.children.subList(0, midIndex));

            for (Node child : newNode.children) {
                child.parent = newNode;
            }
        }

        node.computeMBR();
        newNode.computeMBR();

        if (node == root) {
            root = new Node(node.level + 1);
            root.children.add(node);
            root.children.add(newNode);
            node.parent = root;
            newNode.parent = root;
            root.computeMBR();
        } else {
            newNode.parent = node.parent;
            node.parent.children.add(newNode);
            node.parent.computeMBR();
        }
    }

    private void adjustTree(Node node) {
        if (node == root)
            return;

        Node current = node;
        while (current != root) {
            current.parent.computeMBR();
            if (current.parent.children.size() > MAX_ENTRIES) {
                splitNode(current.parent);
            }
            current = current.parent;
        }
    }

    public List<Point> search(Rectangle queryMBR) {
        return searchNode(root, queryMBR);
    }

    private List<Point> searchNode(Node node, Rectangle queryMBR) {
        List<Point> results = new ArrayList<>();

        if (!node.mbr.overlaps(queryMBR)) {
            return results;
        }

        if (node.isLeaf()) {
            for (Point point : node.points) {
                if (point.x >= queryMBR.minX && point.x <= queryMBR.maxX &&
                        point.y >= queryMBR.minY && point.y <= queryMBR.maxY) {
                    results.add(point);
                }
            }
        } else {
            for (Node child : node.children) {
                results.addAll(searchNode(child, queryMBR));
            }
        }

        return results;
    }
}
