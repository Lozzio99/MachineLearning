package templates;

import algorithms.ANN.Digits.DigitsClassifier;
import algorithms.ANN.XOR.XOR_Solver;
import algorithms.Clustering;

import java.awt.*;

public abstract class Algorithm extends Component {
    private static final long SerialVersionUID = 0L;
    public Algorithm() {
    }
    public abstract void update();

    public abstract void init();
    protected int iteration = 1;
    protected double learningRate = 0.01;


    public static Algorithm getInstance(AlgorithmType type) {
        switch (type) {
            case kMeansClustering -> {
                return new Clustering(AlgorithmType.kMeansClustering.params[0]);
            }
            case ANN_XOR -> {
                return new XOR_Solver(AlgorithmType.ANN_XOR.params);
            }
            case ANN_DIGITS -> {
                return new DigitsClassifier(AlgorithmType.ANN_DIGITS.params);
            }
            default -> {
                throw new IllegalArgumentException();
            }
        }
    }


}
