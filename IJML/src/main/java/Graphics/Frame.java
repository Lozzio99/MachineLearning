package Graphics;

import templates.Algorithm;
import templates.AlgorithmType;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.concurrent.Executors;

import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static templates.AlgorithmType.ANN_XOR;
import static templates.AlgorithmType.kMeansClustering;

public class Frame extends JFrame {

    private final Engine engine;
    public static int width = 1400, height = 800;
    private final AlgorithmType type;

    public Frame(AlgorithmType type) {
        this.type = type;
        this.engine = createEngine();
        this.setWindowProperties();
        this.start();
    }

    private Engine createEngine() {
        Container cp = getContentPane();
        Engine engine = new Engine(Algorithm.getInstance(this.type));
        setSize(new Dimension(width, height));
        engine.setPreferredSize(new Dimension(width, height));
        addKeyListener(new MyKeyAdapter());
        cp.add(engine);
        return engine;
    }
    private void start() {
        long defaultMillis = 100;
        Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(
                this.engine, 1000,(this.type.equals(ANN_XOR)? 100 : (
                        this.type.equals(kMeansClustering)? 500 : defaultMillis
                        )),MILLISECONDS
        );
    }

    private void setWindowProperties() {
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setTitle(engine.toString());
        setResizable(false);
        setVisible(true);
        setLocationRelativeTo(null);// Center window
    }
    private class MyKeyAdapter extends KeyAdapter {
        @Override
        public void keyPressed(KeyEvent keyEvent) {
            if (keyEvent.getKeyCode() == KeyEvent.VK_SPACE) {
                engine.visualization.update();
                repaint();
            }
        }
    }

}
