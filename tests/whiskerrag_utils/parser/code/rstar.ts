class Point {
    constructor(public x: number, public y: number) {}
}

class RStarNode {
    points: Point[];
    children: RStarNode[];
    mbr: { minX: number; minY: number; maxX: number; maxY: number };
    parent: RStarNode | null;
    level: number;

    static readonly MAX_ENTRIES = 4;
    static readonly MIN_ENTRIES = 2;

    constructor(level: number = 0) {
        this.points = [];
        this.children = [];
        this.mbr = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
        this.parent = null;
        this.level = level;
    }

    get isLeaf(): boolean {
        return this.level === 0;
    }

    computeMBR(): void {
        this.mbr.minX = Infinity;
        this.mbr.minY = Infinity;
        this.mbr.maxX = -Infinity;
        this.mbr.maxY = -Infinity;

        if (this.isLeaf) {
            for (const point of this.points) {
                this.mbr.minX = Math.min(this.mbr.minX, point.x);
                this.mbr.minY = Math.min(this.mbr.minY, point.y);
                this.mbr.maxX = Math.max(this.mbr.maxX, point.x);
                this.mbr.maxY = Math.max(this.mbr.maxY, point.y);
            }
        } else {
            for (const child of this.children) {
                this.mbr.minX = Math.min(this.mbr.minX, child.mbr.minX);
                this.mbr.minY = Math.min(this.mbr.minY, child.mbr.minY);
                this.mbr.maxX = Math.max(this.mbr.maxX, child.mbr.maxX);
                this.mbr.maxY = Math.max(this.mbr.maxY, child.mbr.maxY);
            }
        }
    }
}

class RStarTree {
    root: RStarNode;

    constructor() {
        this.root = new RStarNode();
    }

    insert(point: Point): void {
        let node = this.chooseLeaf(this.root, point);
        node.points.push(point);
        node.computeMBR();

        if (node.points.length > RStarNode.MAX_ENTRIES) {
            this.splitNode(node);
        }

        this.adjustTree(node);
    }

    private chooseLeaf(node: RStarNode, point: Point): RStarNode {
        if (node.isLeaf) return node;

        let minIncrease = Infinity;
        let chosen = node.children[0];

        for (const child of node.children) {
            const increase = this.calculateEnlargement(child.mbr, point);
            if (increase < minIncrease) {
                minIncrease = increase;
                chosen = child;
            }
        }

        return this.chooseLeaf(chosen, point);
    }

    private calculateEnlargement(mbr: any, point: Point): number {
        const oldArea = (mbr.maxX - mbr.minX) * (mbr.maxY - mbr.minY);
        const newMinX = Math.min(mbr.minX, point.x);
        const newMinY = Math.min(mbr.minY, point.y);
        const newMaxX = Math.max(mbr.maxX, point.x);
        const newMaxY = Math.max(mbr.maxY, point.y);
        const newArea = (newMaxX - newMinX) * (newMaxY - newMinY);
        return newArea - oldArea;
    }

    private splitNode(node: RStarNode): void {
        const newNode = new RStarNode(node.level);

        if (node.isLeaf) {
            // Sort points by x-coordinate
            node.points.sort((a, b) => a.x - b.x);

            // Move half of the points to the new node
            const midIndex = Math.floor(node.points.length / 2);
            newNode.points = node.points.splice(midIndex);
        } else {
            // Sort children by their MBR's center x-coordinate
            node.children.sort((a, b) => {
                const centerA = (a.mbr.minX + a.mbr.maxX) / 2;
                const centerB = (b.mbr.minX + b.mbr.maxX) / 2;
                return centerA - centerB;
            });

            // Move half of the children to the new node
            const midIndex = Math.floor(node.children.length / 2);
            newNode.children = node.children.splice(midIndex);
            newNode.children.forEach(child => child.parent = newNode);
        }

        // Update MBRs
        node.computeMBR();
        newNode.computeMBR();

        // Handle root split
        if (node === this.root) {
            const newRoot = new RStarNode(node.level + 1);
            newRoot.children.push(node, newNode);
            node.parent = newRoot;
            newNode.parent = newRoot;
            this.root = newRoot;
            newRoot.computeMBR();
        } else {
            newNode.parent = node.parent;
            node.parent!.children.push(newNode);
            node.parent!.computeMBR();
        }
    }

    private adjustTree(node: RStarNode): void {
        if (node === this.root) return;

        let current = node;
        while (current !== this.root) {
            current.parent!.computeMBR();
            if (current.parent!.children.length > RStarNode.MAX_ENTRIES) {
                this.splitNode(current.parent!);
            }
            current = current.parent!;
        }
    }

    search(queryMBR: { minX: number; minY: number; maxX: number; maxY: number }): Point[] {
        return this.searchNode(this.root, queryMBR);
    }

    private searchNode(node: RStarNode, queryMBR: any): Point[] {
        const results: Point[] = [];

        if (!this.overlaps(node.mbr, queryMBR)) {
            return results;
        }

        if (node.isLeaf) {
            return node.points.filter(point =>
                point.x >= queryMBR.minX && point.x <= queryMBR.maxX &&
                point.y >= queryMBR.minY && point.y <= queryMBR.maxY
            );
        }

        for (const child of node.children) {
            results.push(...this.searchNode(child, queryMBR));
        }

        return results;
    }

    private overlaps(mbr1: any, mbr2: any): boolean {
        return !(
            mbr1.maxX < mbr2.minX || mbr1.minX > mbr2.maxX ||
            mbr1.maxY < mbr2.minY || mbr1.minY > mbr2.maxY
        );
    }
}
