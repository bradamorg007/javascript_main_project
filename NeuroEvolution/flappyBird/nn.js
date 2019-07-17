// Can use map function to apply sigmod reulu and tanh functions
// softmax cant use map need to use matrix
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x))
}

function dSigmoid(x) {
    // requires x input to have already been through the sigmoid
    return x * (1 - x);
}

function RELU(x) {
    // Leaky
    return (x < 0) ? 0.01 : x;

}

function dRELU(x) {
    return (x < 0) ? 0.01 : 1;
}

function softmax(x) {
    // x is a the output vector from the outlayer
    // softmax should be only used at the end I think
    if (!x instanceof Matrix2D) {
        throw new Error("Activation Function Error: Softmax must recieve a Matrix2D as input ")
    }

    let minMax_val = x.minMax();

    for (let i = 0; i < x.size().rows; i++) {
        for (let j = 0; j < x.size().cols; j++) {
            x.data[i][j] = x.data[i][j] - minMax_val.max;
        }
    }

    let exps = Matrix2D.map(x, exponential);

    let sumExps = 0;

    for (let i = 0; i < exps.size().rows; i++) {
        for (let j = 0; j < exps.size().cols; j++) {
            sumExps += exps.data[i][j];
        }
    }

    return Matrix2D.divide(exps, sumExps);
}

function dSoftmax(x) {
    // requires x input to have already been through the softmax func. This calcs derivative for a single 
    // logit or yi of the softmax vector
    return x * (1 - x);
}


function exponential(x) {
    return Math.exp(x);
}

function tanh(x) {
    return Math.tanh(x);
}

function dTanh(x) {
    // requires x input to have already been through the tanh
    return 1 - Math.pow(x, 2);
}

function maxMin(x) {

    let max = x[0];
    let min = x[0];

    for (let i = 0; i < x.length; i++) {

        if (x[i] > max) {
            max = x[i];
        } else if (x[i] < min) {
            min = x[i];
        }
    }

    return {
        max: max,
        min: min
    }
}

function mean(x) {

    let sum = 0;

    for (let i = 0; i < x.length; i++) {
        sum += x[i];
    }

    return (1 / x.length) * sum;
}

function sd(x) {
    // standard deviation

    let mean_val = mean(x);
    let z = [];

    for (let i = 0; i < x.length; i++) {
        z.push(Math.pow((x[i] - mean_val), 2));
    }

    return Math.sqrt(mean(z));
}



class NeuralNetwork {

    // Numlayers should include input layer, number of hidden layers and output layers

    constructor(layerUnits, activationFuncList) {

        if (layerUnits === undefined || activationFuncList === undefined) {

            this.numLayers = null;
            // Layers will store the LayerToLayer Objects
            this.layers = null;
            // The number of units per layer
            this.layerUnits = null;
            this.weightInit = null;
            this.activationFunctions = null;
            this.backpropActive = false;


        } else {

            this.numLayers = layerUnits.length;
            // Layers will store the LayerToLayer Objects
            this.layers = [];
            // The number of units per layer
            this.layerUnits = layerUnits;
            this.weightInit = false;
            this.activationFunctions = activationFuncList;

        }

        this.CLASS_TAG = "nn.js";

    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(data) {

        if (typeof data == 'string') {
            data = JSON.parse(data);
        }

        let nn = new NeuralNetwork();

        // Layers will store the LayerToLayer Objects

        let layers = [];
        for (let i = 0; i < data.layers.length; i++) {

            let LayerToLayer = {
                tag: data.layers[i].tag,
                weights: Matrix2D.deserialize(data.layers[i].weights),
                biases: Matrix2D.deserialize(data.layers[i].biases),
                activationFunction: data.layers[i].activationFunction
            }

            layers.push(LayerToLayer);

        }

        // The number of units per layer
        nn.layers = layers;
        nn.numLayers = data.layerUnits.length;
        nn.layerUnits = data.layerUnits;
        nn.weightInit = data.weightInit;
        nn.activationFunctions = data.activationFunctions;

        return nn;
    }

    setLayerUnits(layerUnits) {

        if (NeuralNetwork.layerChecks(layerUnits)) {
            this.layerUnits = layerUnits;
        } else {
            throw new Error("Illegal Argument Exception: Units: Units Must be of type array, not empty or one layer");
        }
    }

    static actFuncCheck(actFuncs, layerUnitsLen) {

        if (actFuncs instanceof Array) {
            if (actFuncs.length === 1) {
                return true;
            } else if (actFuncs.length === (layerUnitsLen - 1)) {
                return true;
            } else {
                return false
            }
        } else {
            return false;
        }
    }
    static layerChecks(units) {

        if (!units instanceof Array) {
            return false;
        } else if (units.length <= 1) {
            return false;
        }

        return true;
    }


    initLayers(initType) {


        if (this.layerUnits == 0) {
            throw new Error("Illegal Argument Exception: The topology of the network has not been defined");
        }


        for (let i = 1; i < this.layerUnits.length; i++) {

            let currentLayer = this.layerUnits[i];
            let previousLayer = this.layerUnits[i - 1];

            let strCurrent = "H" + i;
            let strPrevious = "H" + (i - 1);

            if (i == 1) {
                strPrevious = "in" + (i - 1);
            }

            if (i == this.layerUnits.length - 1) {
                strCurrent = "out" + i;
            }

            let LayerToLayer = {
                tag: strPrevious + "-" + strCurrent,
                weights: new Matrix2D(currentLayer, previousLayer),
                biases: new Matrix2D(currentLayer, 1),
                activationFunction: (this.activationFunctions.length === 1) ? this.activationFunctions[0] : this.activationFunctions[i - 1],
                outputSignal: null
            }

            switch (initType) {

                case "normal":
                    LayerToLayer.weights.normal();
                    LayerToLayer.biases.normal();
                    break;

                case "he_normal":
                    LayerToLayer.weights.he_normal();
                    LayerToLayer.biases.he_normal();
                    break;

                case "xavier":
                    LayerToLayer.weights.xavier_normal();
                    LayerToLayer.biases.xavier_normal();
                    break;

                case "truncated_normal":
                    LayerToLayer.weights.truncated_normal();
                    LayerToLayer.biases.truncated_normal();
                    break;

                case "uniformRandom":
                    LayerToLayer.weights.uniformRandom();
                    LayerToLayer.biases.uniformRandom();
                    break;

            }

            this.layers.push(LayerToLayer);

        }

        this.weightInit = true;
    }


    feedFoward(inputs) {

        inputs = new Matrix2D(inputs.length, 1, inputs);

        for (let i = 0; i < this.layers.length; i++) {

            inputs = Matrix2D.dotProduct(this.layers[i].weights, inputs);
            inputs = Matrix2D.add(inputs, this.layers[i].biases);

            if (this.backpropActive) {
                this.layers[i].outputSignal = inputs;
            }

            switch (this.layers[i].activationFunction) {

                case "sigmoid":
                    inputs = inputs.map(sigmoid);
                    break;

                case "RELU":
                    inputs = inputs.map(RELU);
                    break;

                case "tanh":
                    inputs = inputs.map(tanh);
                    break;

                case "softmax":
                    inputs = softmax(inputs);
                    break;
            }
        }

        return inputs;
    }

    // train(inputs, targets) {

    //     // First 

    // }

    // Cost Functions 

    meanSquaredError(outputs, targets) {

        let results = new Matrix2D(output.size().rows, outputs.size().cols);

        for (let i = 0; i < outputs.size().rows; i++) {
            for (let j = 0; j < outputs.size().cols; j++) {
                results.data[i][j] = Math.pow(targets.data[i][j] - outputs.data[i][j], 2);
            }
        }

        return outputs;
    }

    // derivitaive of mean squared error cost function
    dMeanSquaredError(outputs, targets) {

        let results = new Matrix2D(output.size().rows, outputs.size().cols);

        for (let i = 0; i < outputs.size().rows; i++) {
            for (let j = 0; j < outputs.size().cols; j++) {
                results.data[i][j] = -(targets.data[i][j] - outputs.data[i][j]);
            }
        }

        return outputs;
    }

    // crossEntropyLoss(outputs, targets) {

    //     if (outputs.size().rows !== targets.size().rows || outputs.size().cols !== targets.size().cols)
    //         let sum = 0;

    //     for (let i = 0; i < outputs.size().rows; i++) {
    //         for (let j = 0; j < outputs.size().cols; j++) {

    //             sum += targets.data[i][j] * Math.log(outputs.data[i][j])
    //         }
    //     }

    //     return -(sum);
    // }

    mutate(rate) {

        // Need to go through each layer section - each weight & biases matrix and tweak the
        // the values up or down by a small amount.

        for (let i = 0; i < this.layers.length; i++) {

            // update weights
            for (let j = 0; j < this.layers[i].weights.size().rows; j++) {
                for (let k = 0; k < this.layers[i].weights.size().cols; k++) {

                    if (Math.random() < rate) {
                        //this.layers[i].weights.data[j][k] = Math.random() * 100;
                        this.layers[i].weights.data[j][k] += Matrix2D.gaussian_distribution(0, 1, 1);
                    }

                }
            }

            // Mutate biases 
            for (let l = 0; l < this.layers[i].biases.size().rows; l++) {
                for (let p = 0; p < this.layers[i].biases.size().cols; p++) {

                    if (Math.random() < rate) {
                        //this.layers[i].biases.data[l][p] = Math.random() * 100;
                        this.layers[i].biases.data[l][p] += Matrix2D.gaussian_distribution(0, 1, 1);
                    }
                }
            }

        }

    }


    standardization(x) {

        /**
         * StandardScaler standardizes a feature by subtracting the mean and then scaling to unit variance.
         *  Unit variance means dividing all the values by the standard deviation. StandardScaler does not 
         *  meet the strict definition of scale I introduced earlier.
           StandardScaler results in a distribution with a standard deviation equal to 1. The variance is 
           equal to 1 also, because variance = standard deviation squared. And 1 squared = 1.
           StandardScaler makes the mean of the distribution 0. About 68% of the values will lie be between -1 and 1.
         */

        let mean_val = mean(x);
        let sd_val = sd(x);
        let output = [];

        for (let i = 0; i < x.length; i++) {
            output.push((x[i] - mean_val) / sd_val);
        }

        return output;
    }

    normalise(x) {

        /**
         * For each value in a feature, MinMaxScaler subtracts the minimum value in the feature and then divides by the range. 
         * The range is the difference between the original maximum and original minimum.
           MinMaxScaler preserves the shape of the original distribution. It doesn’t meaningfully change the information embedded in the original data.
           Note that MinMaxScaler doesn’t reduce the importance of outliers.
           The default range for the feature returned by MinMaxScaler is 0 to 1.
         */

        let minMax_val = maxMin(x);

        let max = minMax_val.max;
        let min = minMax_val.min;
        let output = [];

        for (let i = 0; i < x.length; i++) {
            output.push((x[i] - min) / (max - min))
        }

        return output;
    }



}

if (typeof module !== 'undefined') {
    module.exports = NeuralNetwork;
}