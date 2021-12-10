package templates;

public enum AlgorithmType {
    kMeansClustering(5),
    ANN_XOR(2,3,128,1,true),
    ANN_DIGITS(400,2,48,10,true);
    Object[] params;
    AlgorithmType(Object... args){
        this.params = args;
    }
}
