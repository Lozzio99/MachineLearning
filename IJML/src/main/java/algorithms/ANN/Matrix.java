package algorithms.ANN;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import static java.lang.System.arraycopy;

/**
 * The type Matrix.
 */
public class Matrix implements Serializable {
    /**
     * The N rows.
     */
    public int n_rows;
    /**
     * The N cols.
     */
    public int n_cols;
    /**
     * The Matrix.
     */
    public final double[][] matrix;

    /**
     * Constructor from array
     *
     * @param input the array resulting in a row
     * @return a single column matrix
     */
    public Matrix(double[] input) {
        this.matrix = new double[input.length][1];
        for (int i = 0; i < input.length; i++)
            this.matrix[i][0] = input[i];
        this.n_rows = input.length;
        this.n_cols = 1;
    }

    /**
     * Constructor with null matrix
     *
     * @param num_rows the number of rows
     * @param num_cols the number of columns
     * @return a null matrix with given dimensions
     */
    public Matrix(int num_rows, int num_cols) {
        this.n_cols = num_cols;
        this.n_rows = num_rows;
        this.matrix = new double[num_rows][num_cols];
    }

    /**
     * @param m the given matrix
     * @return the new matrix with the  of the old one
     */
    public static  Matrix dfunc( Matrix m) {
        double[][] result = new double[m.n_rows][m.n_cols];
        for (int i = 0; i < m.n_rows; i++) {
            for (int k = 0; k < m.n_cols; k++) {
                result[i][k] = m.getMatrix()[i][k] * (1 - m.getMatrix()[i][k]);
            }
        }
        return new Matrix(result);
    }

    /**
     * mathod for the activation function
     *
     * @param x the value to activate
     * @return the value activated
     */
    public static double activation(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * static method that performs the activation on a matrix
     *
     * @param m the given matrix
     * @return a new matrix after each value has been activated
     */
    public static  Matrix map( Matrix m) {
        double[][] result = new double[m.n_rows][m.n_cols];
        for (int i = 0; i < m.n_rows; i++) {
            for (int k = 0; k < m.n_cols; k++) {
                result[i][k] = activation(m.getMatrix()[i][k]);
            }
        }
        return new Matrix(result);
    }

    /**
     * Constructor from array
     *
     * @param matrix the double dimensions array Construct a matrix with the given values
     */
    public Matrix(double[]  [] matrix) {
        this.n_rows = matrix.length;
        this.n_cols = matrix[0].length;
        this.matrix = matrix;
    }

    /**
     * static method for the matrix multiplication
     *
     * @param A the left side matrix
     * @param B the right side matrix
     * @return the product from the two in a matrix
     */
    public static  Matrix multiply( Matrix A,  Matrix B) {
        if (A.n_cols != B.n_rows)
            throw new IllegalArgumentException("Columns of A and rows  and B must have same dimensions!");

        double[][] res = new double[A.n_rows][B.n_cols];
        for (int i = 0; i < A.n_rows; i++) {
            for (int j = 0; j < B.n_cols; j++) {
                double sum = 0;
                for (int k = 0; k < A.n_cols; k++) {
                    sum += A.getMatrix()[i][k] * B.getMatrix()[k][j];
                }
                res[i][j] = sum;
            }
        }
        return new Matrix(res);
    }

    /**
     * static method that performs a randomization of a matrix
     *
     * @param m the old matrix
     * @return a new Matrix randomized from -1 to 1
     */
    public static  Matrix randomize( Matrix m) {
        double[][] res = new double[m.n_rows][m.n_cols];
        for (int i = 0; i < m.n_rows; i++) {
            for (int k = 0; k < m.n_cols; k++) {
                res[i][k] = new Random().nextDouble();
                if (new Random().nextDouble() < 0.5) {
                    res[i][k] *= -1;
                }
            }
        }
        return new Matrix(res);
    }

    /**
     * static method to perform a matrix subtraction
     *
     * @param n the left side matrix
     * @param m the right side matrix
     * @return a new matrix after the subtraction
     */
    public static  Matrix subtract( Matrix n,  Matrix m) {
        if (n.n_rows != m.n_rows || n.n_cols != m.n_cols) {
            throw new IllegalArgumentException("Columns and rows of A and B must have same dimensions!");
        }
        double[][] result = new double[n.n_rows][n.n_cols];
        for (int i = 0; i < n.n_rows; i++) {
            for (int k = 0; k < n.n_cols; k++) {
                result[i][k] = n.getMatrix()[i][k] - m.getMatrix()[i][k];
            }
        }
        return new Matrix(result);
    }

    /**
     * static method to perform a matrix addition
     *
     * @param n the left side matrix
     * @param m the right side matrix
     * @return a new matrix after the addition
     */
    public static  Matrix add( Matrix n,  Matrix m) {
        if (n.n_rows != m.n_rows || n.n_cols != m.n_cols) {
            throw new IllegalArgumentException("Columns and rows of A and B must have same dimensions!");
        }
        double[][] res = new double[n.n_rows][n.n_cols];
        for (int i = 0; i < n.n_rows; i++) {
            for (int k = 0; k < n.n_cols; k++) {
                res[i][k] = n.getMatrix()[i][k] + m.getMatrix()[i][k];
            }
        }
        return new Matrix(res);
    }

    /**
     * LA matrix transpose
     * a new matrix equal is setted but transposed
     */
    public static Matrix transpose(Matrix m) {
        Matrix result = new Matrix(m.n_cols, m.n_rows);
        for (int i = 0; i < m.n_cols; i++) {
            for (int k = 0; k < m.n_rows; k++) {
                result.matrix[i][k] = m.getMatrix()[k][i];
            }
        }
        return result;
    }

    /**
     * instance method to multiply a matrix by a single value
     *
     * @param value the value to add
     * @return a new Matrix after the evaluation
     */
    public Matrix multiply(double value) {
        double[][] res = new double[this.n_rows][this.n_cols];
        for (int i = 0; i < this.n_rows; i++) {
            for (int k = 0; k < this.n_cols; k++) {
                res[i][k] = this.getMatrix()[i][k] * value;
            }
        }
        return new Matrix(res);
    }

    /**
     * instance method for the dot product
     *
     * @param m the right sided matrix
     * @return a new matrix resulting from the dot product
     */
    public Matrix multiply(Matrix m) {
        if (m.n_rows == 1 && m.n_cols == 1) {
            return this.multiply(m.getMatrix()[0][0]);
        }
        if (this.n_rows != m.n_rows || this.n_cols != m.n_cols) {
            throw new IllegalArgumentException("Columns and rows of A and B must have same dimensions!");
        }
        double[][] res = new double[this.n_rows][m.n_cols];
        for (int i = 0; i < this.n_rows; i++) {
            for (int j = 0; j < this.n_cols; j++) {
                res[i][j] = this.getMatrix()[i][j] * m.getMatrix()[i][j];
            }
        }
        return new Matrix(res);
    }

    /**
     * Invert double [ ] [ ].
     *
     * @param in the in
     * @return the double [ ] [ ]
     */
    public static double[][] inverse(double[][] in) {
        if (in.length != in[0].length)
            throw new IllegalArgumentException("Can't invert non square matrices");
        double[][] a = new double[in.length][in[0].length];

        for (int i = 0; i < in.length; i++) {
            arraycopy(in[i],
                    0,
                    a[i],
                    0,
                    in[0].length);
        }
        int N = a.length;

        double[][] inverse = new double[N][N], b = new double[N][N];
        int[] indexes = new int[N];

        for (int x = 0; x < N; x++) b[x][x] = 1;
        gaussian(a, indexes); // Transform the matrix into an upper triangle
        // Update the matrix b[v][u] with the ratios stored
        for (int v = 0; v < N - 1; v++)
            for (int u = v + 1; u < N; u++)
                for (int w = 0; w < N; w++) {
                    b[indexes[u]][w] -= a[indexes[u]][v] * b[indexes[v]][w];
                }

        for (int k = 0; k < N; k++) {
            inverse[N - 1][k] = (b[indexes[N - 1]][k]) / (a[indexes[N - 1]][N - 1]);
            for (int j = N - 2; j >= 0; j--) {
                inverse[j][k] = b[indexes[j]][k];        // Perform backward substitutions
                for (int l = j + 1; l < N; l++) {
                    inverse[j][k] -= a[indexes[j]][l] * inverse[l][k];
                }
                inverse[j][k] /= a[indexes[j]][j];
            }
        }
        return inverse;
    }

    public void setMatrix(double[][] matrix) {
        IntStream.range(0, matrix.length).forEachOrdered(i -> this.matrix[i] = matrix[i]);
        this.n_cols = matrix.length;
        this.n_rows = matrix[0].length;
    }

    /**
     * Gaussian.
     *
     * @param m     the matrix
     * @param index the index
     */
    public static void gaussian(double[][] m, int[] index) {
        int n = index.length;
        double[] c = new double[n];

        // Initialize the index
        for (int i = 0; i < n; ++i)
            index[i] = i;

        // Find the rescaling factors, one from each row
        for (int i = 0; i < n; ++i) {
            double c1 = 0;
            for (int j = 0; j < n; ++j) {
                double c0 = Math.abs(m[i][j]);
                if (c0 > c1) c1 = c0;
            }
            c[i] = c1;
        }

        // Search the pivoting element from each column
        int k = 0;
        for (int j = 0; j < n - 1; ++j) {
            double pi1 = 0;
            for (int i = j; i < n; ++i) {
                double pi0 = Math.abs(m[index[i]][j]);
                pi0 /= c[index[i]];
                if (pi0 > pi1) {
                    pi1 = pi0;
                    k = i;
                }
            }

            // Interchange rows according to the pivoting order
            int itmp = index[j];
            index[j] = index[k];
            index[k] = itmp;
            for (int i = j + 1; i < n; ++i) {
                double pj = m[index[i]][j] / m[index[j]][j];

                // Record pivoting ratios below the diagonal
                m[index[i]][j] = pj;

                // Modify other elements accordingly
                for (int l = j + 1; l < n; ++l)
                    m[index[i]][l] -= pj * m[index[j]][l];
            }
        }
    }
    /**
     * accessor method
     *
     * @return the matrix of this field
     */
    public double[][] getMatrix() {
        return this.matrix;
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();

        for (double[] m : this.matrix) s.append(Arrays.toString(m)).append("\n");

        return "Matrix{ " + n_rows +
                "x" + n_cols +
                ",\n" + s.toString().trim() +
                '}';
    }

    public void printMatrix() {
        for (int i = 0; i < this.n_rows; i++) {
            for (int k = 0; k < this.n_cols; k++) {
                System.out.print(this.matrix[i][k] + " ");
            }
            System.out.println();
        }
    }

}
