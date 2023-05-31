import java.util.ArrayList;
import java.util.List;

class Point {
    private double x;
    private double y;
    private int cluster;

    public Point(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public int getCluster() {
        return cluster;
    }

    public void setCluster(int cluster) {
        this.cluster = cluster;
    }
}

class Cluster {
    private double centroidX;
    private double centroidY;
    private List<Point> points;

    public Cluster(double centroidX, double centroidY) {
        this.centroidX = centroidX;
        this.centroidY = centroidY;
        this.points = new ArrayList<>();
    }

    public double getCentroidX() {
        return centroidX;
    }

    public double getCentroidY() {
        return centroidY;
    }

    public List<Point> getPoints() {
        return points;
    }

    public void addPoint(Point point) {
        points.add(point);
    }

    public void clear() {
        points.clear();
    }

    public void updateCentroid() {
        double sumX = 0;
        double sumY = 0;

        for (Point point : points) {
            sumX += point.getX();
            sumY += point.getY();
        }

        if (!points.isEmpty()) {
            centroidX = sumX / points.size();
            centroidY = sumY / points.size();
        }
    }
}

public class KMeans {
    private List<Point> points;
    private List<Cluster> clusters;
    private int k;

    public KMeans(List<Point> points, int k) {
        this.points = points;
        this.k = k;
        this.clusters = new ArrayList<>();
    }

    public void initialize() {
        for (int i = 0; i < k; i++) {
            Cluster cluster = new Cluster(points.get(i).getX(), points.get(i).getY());
            clusters.add(cluster);
        }
    }

    public void run() {
        boolean converged = false;

        while (!converged) {
            // Assign points to the nearest cluster
            for (Point point : points) {
                int nearestCluster = 0;
                double minDistance = distance(point, clusters.get(0));

                for (int i = 1; i < k; i++) {
                    double distance = distance(point, clusters.get(i));
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearestCluster = i;
                    }
                }

                point.setCluster(nearestCluster);
                clusters.get(nearestCluster).addPoint(point);
            }

            // Update centroids
            converged = true;
            for (Cluster cluster : clusters) {
                double prevCentroidX = cluster.getCentroidX();
                double prevCentroidY = cluster.getCentroidY();
                cluster.updateCentroid();
                double currentCentroidX = cluster.getCentroidX();
                double currentCentroidY = cluster.getCentroidY();

                if (prevCentroidX != currentCentroidX || prevCentroidY != currentCentroidY) {
                    converged = false;
                    cluster.clear();
                }
            }
        }
    }

    private double distance(Point point, Cluster cluster) {
        double dx = point.getX() - cluster.getCentroidX();
        double dy = point.getY() - cluster.getCentroidY();
        return Math.sqrt(dx * dx + dy * dy);
    }

    public static void main(String[] args) {
        List<Point> points = new ArrayList<>();
        points.add(new Point(1.0, 1.0));
        points.add(new Point(1.5, 2.0));
        points.add(new Point(3.0, 4.0));
        points.add(new Point(5.0, 7.0));
        points.add(new Point(3.5, 5.0));
        points.add(new Point(4.5, 5.0));

        int k = 2;
        KMeans kMeans = new KMeans(points, k);
        kMeans.initialize();
        kMeans.run();

        for (Cluster cluster : kMeans.clusters) {
            System.out.println("Cluster centroid: (" + cluster.getCentroidX() + ", " + cluster.getCentroidY() + ")");
            System.out.println("Points in this cluster:");
            for (Point point : cluster.getPoints()) {
                System.out.println("(" + point.getX() + ", " + point.getY() + ")");
            }
            System.out.println();
        }
    }
}
