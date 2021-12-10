package algorithms.ANN.XOR;

import algorithms.ANN.ANN_Data;
import algorithms.ANN.ArtificialNeuralNetwork;
import algorithms.ANN.Matrix;

import java.awt.*;

import static Graphics.Frame.*;
public class XOR_Solver extends ArtificialNeuralNetwork {
    private static final int resolution = 10,
            cols = width / resolution,
            rows = height / resolution;
    private static Matrix[] in, tg;
    private static ANN_Data[] tests;

    public XOR_Solver(Object[] params) {
        super(params);
    }
    @Override
    public void init() {
        initData();
        this.setData(in,tg);
    }

    public static void initData() {
        tests = new ANN_Data[4];
        generateXORPoints();
        in = new Matrix[tests.length];
        tg = new Matrix[tests.length];
        for (int i = 0; i < tests.length; i++) {
            in[i] = tests[i].inputMatrix();
            tg[i] = tests[i].outputMatrix();
        }
    }
    private static void generateXORPoints() {
        tests[0] = new ANN_Data(new double[]{1, 1}, new double[]{0});
        tests[1] = new ANN_Data(new double[]{1, 0}, new double[]{1});
        tests[2] = new ANN_Data(new double[]{0, 1}, new double[]{1});
        tests[3] = new ANN_Data(new double[]{0, 0}, new double[]{0});
    }

    @Override
    public void update() {
        if (iteration % 100 == 0) System.out.println(iteration + "  - learning rate : " + learningRate);
        super.update();
        iteration++;
        if (learningRate <= 0.4) lk = (learningRate += 1e-6);
        this.repaint();
    }

    @Override
    public String toString() {
        return "XOR PROBLEM";
    }

    public void paint(Graphics g) {
        Graphics2D g2 = (Graphics2D) g;
        for (int i = 0; i < cols; i++) {
            for (int k = 0; k < rows; k++) {
                double x1 = i / (double) cols;
                double x2 = k / (double) rows;
                double[] inputs = new double[]{x1, x2};
                int y = (int) (this.guess(inputs).getMatrix()[0][0] * 255);
                Color r = new Color(255 - y, y, y / 2);
                g2.setColor(r);
                g2.fillRect(i * resolution, k * resolution, resolution, resolution);
                g2.setColor(Color.white);
                g2.drawRect(i * resolution, k * resolution, resolution, resolution);
            }
        }

    }


}
