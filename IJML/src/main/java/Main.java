import Graphics.Frame;
import templates.AlgorithmType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Main {
    private static final long SerialVersionUID = 0L;

    public Main() {
    }

    public static void main(String[] args) {
        //new Frame(AlgorithmType.ANN_XOR);
        //new Frame(AlgorithmType.kMeansClustering);
        new Frame(AlgorithmType.ANN_DIGITS);
    }
}
