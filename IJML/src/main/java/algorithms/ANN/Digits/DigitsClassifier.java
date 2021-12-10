package algorithms.ANN.Digits;

import algorithms.ANN.ANN_Data;
import algorithms.ANN.ArtificialNeuralNetwork;
import algorithms.ANN.Matrix;

import static Graphics.Frame.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.List;

public class DigitsClassifier extends ArtificialNeuralNetwork {
    private static final long SerialVersionUID = 0L;
    private static Matrix[] testX;
    private static Matrix[] testY;

    public DigitsClassifier(Object[] params) {
        super(params);
    }
    @Override
    public void update() {
        if (iteration % 100 == 0) {
            System.out.print("iteration " + iteration  + "\t");
            testPrediction();
        }
        super.update();
        iteration++;
        if (learningRate <= 0.4) lk = (learningRate += 1e-6);

    }
    private void testPrediction() {
        int rand = new Random().nextInt(testX.length);
        Matrix mx = testX[rand], my = Matrix.transpose(testY[rand]);
        int target = interpretOutput(my.getMatrix()[0]) ;
        double [] in = Matrix.transpose(mx).getMatrix()[0];
        double [] out = Matrix.transpose(guess(in)).getMatrix()[0];
        int output = interpretOutput(out);
        System.out.println("Predicting : " + target + " as -> "+ output);
    }
    @Override
    public void init() {
        List<ANN_Data> data = new ArrayList<>();
        File f = new File(Objects.requireNonNull(DigitsClassifier.class.getClassLoader().getResource("digits/digits.csv")).getFile());
        try (BufferedReader br = new BufferedReader (new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] x = new double[values.length-1];
                int y;
                for(int i = 0; i<values.length-1; i++){
                    x[i] = Double.parseDouble(values[i]);
                }
                y = (int)(Double.parseDouble(values[values.length-1]) % 10);
                data.add(new ANN_Data(x,createOutput(y)));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        Collections.shuffle(data);
        int l = data.size();
        int test_set_size = (l / 100) * 33,
                train_set_size = l-test_set_size;

        Matrix[] trainX = new Matrix[train_set_size];
        Matrix[] trainY = new Matrix[train_set_size];
        testX = new Matrix[test_set_size];
        testY = new Matrix[test_set_size];

        int i = 0;
        for (ANN_Data d : data){
            if (i < train_set_size){
                trainX[i] = d.inputMatrix();
                trainY[i] = d.outputMatrix();
            }else {
                testX[i - train_set_size] = d.inputMatrix();
                testY[i - train_set_size] = d.outputMatrix();
            }
            i++;
        }
        //showSamples();
        setData(trainX, trainY);
    }
    private static Map.Entry<Integer,BufferedImage> randomSample;

    @Override
    public void paint(Graphics g) {
        if (randomSample != null) {
            super.paint(g);
            Frame.getFrames()[0].setTitle("DISPLAYING DIGIT :"+ randomSample.getKey());
            g.setClip(200,200, width, height);
            Image resultingImage = randomSample.getValue().getScaledInstance(400, 400, Image.SCALE_DEFAULT);
            g.drawImage(resultingImage, 200, 200, null);
        }
        if (iteration % 50 == 0){
            randomSample = getRandomSample();
        }

    }
    private double[] createOutput(int y) {
        double[] out = new double[10];
        out[y] = 1;
        return out;
    }
    private static int interpretOutput(double[] out) {
        int idx = -1;
        double max = Integer.MIN_VALUE;
        for (int i = 0; i< out.length; i++){
            if (out[i] > max){
                max = out[i];
                idx = i;
            }
        }
        return idx%10;
    }

    private static Map.Entry<Integer, BufferedImage> getRandomSample() {
        BufferedImage img = new BufferedImage(20,20,BufferedImage.TYPE_BYTE_GRAY);
        int rand = new Random().nextInt(testX.length);
        double[] mx = Matrix.transpose(testX[rand]).getMatrix()[0],
                my = Matrix.transpose(testY[rand]).getMatrix()[0];
        for (int y = 0; y < 20; y++) {
            for (int x = 0; x < 20; x++) {
                int v = map(mx[y + (20 * x)]) ;
                img.setRGB(x, y, new Color(v,v,v).getRGB());
            }
        }
        //JOptionPane.showMessageDialog(null, new JLabel(new ImageIcon(img)));
        /*
        try {
            String path = Objects.requireNonNull(DigitsClassifier.class.getClassLoader().getResource("digits")).getFile();
            ImageIO.write(img, "png",   new File( path+ "/"+(int)my[0]%10+".png"));
        } catch (IOException ex) {
            ex.printStackTrace();
        }
         */
        return Map.entry(interpretOutput(my),img);
    }

    private static int map(double x) {
        double in_min = -0.1320;
        double in_max = 1.1277;
        double out_min = 0;
        double out_max = 255;
        return (int) ((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min);
    }


}
