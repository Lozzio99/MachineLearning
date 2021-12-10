package Graphics;

import templates.Algorithm;

import javax.swing.*;
import java.awt.*;

class Engine extends JPanel implements Runnable {
    public Algorithm visualization;

    public Engine(Algorithm visualization) {
        this.visualization = visualization;
        this.visualization.init();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.setColor(new Color(61, 58, 58));
        g.fill3DRect(0, 0, Frame.width, Frame.height, false);
        visualization.paint(g);
    }

    @Override
    public void run() {
        this.visualization.update();
        this.repaint();
    }

    @Override
    public String toString() {
        return visualization.toString();
    }
}
