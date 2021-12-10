package algorithms.ANN;

import java.util.Arrays;

public record ANN_Data(double[] input, double[] output) {

    public Matrix inputMatrix() {
        return new Matrix(input);
    }

    public Matrix outputMatrix() {
        return new Matrix(output);
    }

    @Override
    public String toString() {
        return "{ (" + input.length +
                " x 1 ) => " + Arrays.toString(output) + '}';
    }
}
