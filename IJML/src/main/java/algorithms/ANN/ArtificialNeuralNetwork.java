package algorithms.ANN;

import templates.Algorithm;

import java.awt.*;
import java.util.Random;

public class ArtificialNeuralNetwork extends Algorithm {
    protected Matrix[] W, b; // weights, biases
    public double lk = 0.1;
    private Matrix[] input, target;
    public ArtificialNeuralNetwork(int in, int h_layers, int h_nodes, int out, boolean random) {
        this.W = new Matrix[h_layers +2];  // in - h - out
        this.b = new Matrix[h_layers +2];
        this.W[0] = new Matrix(h_nodes, in);
        this.b[0] = new Matrix(h_nodes, 1);

        for (int i =0; i< h_layers; i++){
            this.W[1+i] = new Matrix(h_nodes, h_nodes);
            this.b[1+i] = new Matrix(h_nodes, 1);
        }

        this.W[this.W.length-1] = new Matrix(out, h_nodes);
        this.b[this.b.length-1] = new Matrix(out, 1);
        if (random)  this.initRandomWeights();
    }

    public ArtificialNeuralNetwork(Object[] params) {
        this(
                (int) params[0],
                (int) params[1],
                (int) params[2],
                (int) params[3],
                (boolean) params[4]
        );
    }

    private void initRandomWeights() {
        for (int i = 0; i< this.W.length; i++) {
            this.W[i] = Matrix.randomize(this.W[i]);
            this.b[i] = Matrix.randomize(this.b[i]);
        }
    }
    public Matrix[] feedForward(final Matrix inputs) {
        Matrix[] res = new Matrix[this.W.length];
        //generating input layer output
        Matrix hidden1 = Matrix.multiply(this.W[0], inputs);
        hidden1 = Matrix.add(hidden1, this.b[0]);
        hidden1 = Matrix.map(hidden1);
        res[0] = hidden1;
        //generating hidden layer output
        Matrix hidden2 = hidden1;

        for (int i = 0; i< this.W.length-2; i++){
            hidden2 = Matrix.multiply(this.W[i+1], hidden2);
            hidden2 = Matrix.add(hidden2, this.b[i+1]);
            hidden2 = Matrix.map(hidden2);
            res[i+1] = hidden2;
        }

        //generating final output
        Matrix outputs = Matrix.multiply(this.W[this.W.length-1], hidden2);
        outputs = Matrix.add(outputs, this.b[this.b.length-1]);
        outputs = Matrix.map(outputs);
        res[res.length-1] = outputs;
        return res;
    }
    public Matrix[] stochasticGradientDescend(final Matrix outError, final Matrix out, final Matrix in) {
        //calculate hidden gradient
        Matrix h1_gradient = Matrix.dfunc(out);
        h1_gradient = h1_gradient.multiply(outError);
        h1_gradient = h1_gradient.multiply(lk);
        //calculate and adjust hidden W
        Matrix inputs_T = Matrix.transpose(in);
        Matrix d_ih = Matrix.multiply(h1_gradient, inputs_T);
        return new Matrix[]{ h1_gradient, d_ih, outError };
    }
    public void train(Matrix given, Matrix targets) {
        Matrix[] prediction = feedForward(given);
        Matrix tgt = new Matrix(targets.getMatrix());

        //calculate output layer error
        Matrix outError = Matrix.subtract(tgt, prediction[prediction.length-1]);
        Matrix[] gradient = stochasticGradientDescend(outError,
                prediction[prediction.length-1],
                prediction[prediction.length-2]
        );
        this.W[this.W.length-1] = Matrix.add(this.W[this.W.length-1], gradient[1]);
        this.b[this.b.length-1] = Matrix.add(this.b[this.b.length-1], gradient[0]);


        for (int i = this.W.length-2; i>0; i--){
            //calculate hidden layers error
            tgt = Matrix.transpose(this.W[i+1]);
            outError = Matrix.multiply(tgt, outError);
            gradient = stochasticGradientDescend(outError,
                    prediction[i],
                    prediction[i-1]
            );
            this.W[i] = Matrix.add(this.W[i], gradient[1]);
            this.b[i] = Matrix.add(this.b[i], gradient[0]);
        }

        //calculate input layer error
        tgt = Matrix.transpose(this.W[1]);
        outError = Matrix.multiply(tgt, gradient[2]);
        gradient = stochasticGradientDescend(
                outError,
                prediction[0],
                given
        );

        this.W[0] = Matrix.add(this.W[0], gradient[1]);
        this.b[0] = Matrix.add(this.b[0], gradient[0]);
    }

    public void train() {
        Random r = new Random();
        this.train(this.input[r.nextInt(this.input.length-1)],
                this.target[r.nextInt(this.target.length-1)]);

    }
    public void setData(Matrix[] inputs, Matrix[] targets){
        if (inputs.length!= targets.length) throw new IllegalArgumentException();
        this.input = inputs;
        this.target = targets;
    }
    public Matrix guess(final double[] inputs) {
        Matrix[] x =  feedForward(new Matrix(inputs));
        return x[x.length-1];
    }
    @Override
    public void update() {
        this.train();
    }

    @Override
    public void init() {

    }

}
