package algorithms;

import templates.Algorithm;

import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

import static java.lang.Math.abs;
import static java.lang.Math.pow;
import static Graphics.Frame.*;

public class Clustering extends Algorithm {
    private final static int SAMPLE_SIZE = 2000;
    private static final Map<Integer, Color> colors = Map.of(
            0,new Color(255, 1, 1),
            1,new Color(255, 131, 34),
            2,new Color(255, 230, 48),
            3,new Color(154, 255, 30),
            4,new Color(22, 255, 104),
            5,new Color(39, 252, 255),
            6,new Color(22, 124, 255),
            7,new Color(29, 0, 213),
            8,new Color(184, 6, 255),
            9,new Color(255, 24, 209)
    );
    private final static int MAX_CLUSTERS = colors.size();
    private static int CLUSTERS;
    private static boolean CONVERGED = false;


    private int step;
    private Map<Element, List<Element>> clusters;
    private final Element[] elements, centroids;
    private final int [] size;
    private final double[] avgX, avgY;


    public Clustering(Object... param) {
        if (param.length>0) {
            if (param[0] instanceof Integer i)
                CLUSTERS = i;
            else throw new IllegalArgumentException();
        }
        else CLUSTERS = MAX_CLUSTERS;

        elements = new Element[SAMPLE_SIZE];
        centroids = new Element[CLUSTERS];
        size = new int[CLUSTERS];
        avgX = new double[CLUSTERS];
        avgY = new double[CLUSTERS];
    }

    @Override
    public void paint(Graphics gr) {
        Graphics2D g = (Graphics2D)gr;
        int sz = 8, s = sz/2;
        if (CONVERGED){
            for (int i = 0; i< CLUSTERS; i++){
                g.setColor(colors.get(centroids[i].kC));
                for (Element e : clusters.get(centroids[i])){
                    g.drawLine((int)centroids[i].x, (int)centroids[i].y, (int)e.x, (int)e.y);
                    g.fillOval((int)e.x-s,(int)e.y-s,sz,sz);
                }
            }
        } else {
            for (Element e : elements) {
                if (e.kC == -1){
                    g.setColor(new Color(
                            mapX(e.x),
                            0,
                            mapY(e.y)
                    ));
                }
                else g.setColor(colors.get(e.kC));
                g.fillOval((int)e.x-s,(int)e.y-s,sz,sz);
            }
            for (int i = 0; i< CLUSTERS; i++){
                if (centroids[i] == null) continue;
                g.setColor(Color.WHITE);
                g.drawLine((int)centroids[i].x-20, (int)centroids[i].y, (int)centroids[i].x+20, (int)centroids[i].y);
                g.drawLine((int)centroids[i].x, (int)centroids[i].y-20, (int)centroids[i].x, (int)centroids[i].y+20);
            }
        }

    }

    private int mapY(double v) {
        return (int) (v * 255 / height);
    }

    private int mapX(double v){
        return (int) (v * 255 / width);
    }

    @Override
    public void update() {
        if (CONVERGED) return;
        switch (step) {
            case 0 -> {
                // INITIALIZE RANDOM CLUSTERS
                List<Integer> v = new ArrayList<>();
                for (int i = 0; i< SAMPLE_SIZE; i++) v.add(i);
                Collections.shuffle(v);
                for (int k = 0; k< CLUSTERS; k++){
                    Element r = elements[v.get(k)];
                    r.kC = k;
                    centroids[k] = r;
                }
            }
            case 1 -> {
                // ASSIGN ELEMENTS TO CLOSEST CLUSTER
                for (Element e : elements) {
                    double minDist = Integer.MAX_VALUE;
                    Element closestCluster = null;
                    for (Element c : centroids){
                        double d = c.distance(e);
                        if (d < minDist){
                            closestCluster = c;
                            minDist = d;
                        }
                    }
                    e.kC = Objects.requireNonNull(closestCluster).kC;
                }
            }
            case 2 -> {
                boolean converged = true;
                // calculate and re-initialize mean of existing centroids as new centroids
                for (int i = 0; i < CLUSTERS; i++) {
                    avgX[i] = avgY[i] = size[i] = 0;
                    for (Element e : elements) {
                        if (e.kC == i){
                            avgX[i] += e.x;
                            avgY[i] += e.y;
                            size[i] ++ ;
                        }
                    }
                    Element ck = new Element(avgX[i] /  size[i],avgY[i] / size[i], centroids[i].kC);
                    if (!ck.similar(centroids[i])) {
                        converged = false;
                    }
                    centroids[i] = ck;
                }
                if (converged) buildMap();
            }

        }
        step = step == 2 ? 1 :  step+1;

    }


    private void buildMap() {
        System.out.println("CONVERGED");
        CONVERGED = true;
        clusters = new HashMap<>();
        for (Element c : centroids) {
            List<Element> list = Arrays.stream(elements).filter(e -> e.kC == c.kC).collect(Collectors.toList());
            clusters.put(c, list);
        }
        System.out.println(clusters);
    }

    @Override
    public void init() {
        for (int i = 0; i< SAMPLE_SIZE; i++){
            elements[i] = new Element(
                    new Random().nextInt(width-5),
                    new Random().nextInt(height-5),
                    -1
            );
        }

    }

    @Override
    public String toString() {
        return "Clustering";
    }

    private final static class Element {
        private final double x;
        private final double y;
        private int kC;

        private Element(double x , double y, int kC){
           this.x = x;
           this.y = y;
           if (kC > MAX_CLUSTERS) throw new IllegalArgumentException();
           this.kC = kC;
        }

        private double distance(Element e){
            return pow(e.x - this.x, 2) + pow(e.y - this.y, 2);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Element e = (Element) o;
            return this.x == e.x && this.y == e.y && this.kC == e.kC;
        }

        public boolean similar(Element e){
            return abs(this.x - e.x) < 1e-4 &&
                    abs(this.y - e.y) < 1e-4 ;
        }

        @Override
        public String toString() {
            return "E" +kC+
                    "{" + x + ", " + y + "}";
        }
    }
}
