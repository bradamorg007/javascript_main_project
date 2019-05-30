// Can use map function to apply sigmod reulu and tanh functions
// softmax cant use map need to use matrix
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x))
}

function RELU(x) {
    // Leaky
    return (x < 0) ? 0.01 : x;

}

function softmax(x) {
    // x is a the output vector from the outlayer
    // softmax should be only used at the end I think
    if (!x instanceof Matrix2D) {
        throw new Error("Activation Function Error: Softmax must recieve a Matrix2D as input ")
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


function exponential(x) {
    return Math.exp(x);
}

function tanh(x) {
    return Math.tanh(x);
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

        this.initTypes = ["normal", "he_normal", "xavier", "truncated_normal"];
        this.activationFunctions = ["sigmoid", "RELU", "tanh", "softmax"];
        // +2 because all networks have 

        if (NeuralNetwork.layerChecks(layerUnits)) {
            this.numLayers = layerUnits.length;

            if (NeuralNetwork.actFuncCheck(activationFuncList, layerUnits.length)) {
                // Layers will store the LayerToLayer Objects
                this.layers = [];
                // The number of units per layer
                this.layerUnits = layerUnits;
                this.weightInit = false;

                let match = false;
                for (let i = 0; i < activationFuncList.length; i++) {
                    for (let j = 0; j < this.activationFunctions.length; j++) {

                        if (activationFuncList[i] === this.activationFunctions[j]) {
                            match = true;
                            break;
                        }
                    }

                    if (match) {
                        if (activationFuncList.length === 1) {
                            break;
                        }
                    } else {
                        throw new Error("Illegal Argument Exception: Actication Function list contains an invalid entry");
                    }
                }

                this.activationFunctions = activationFuncList;
            } else {
                throw new Error("Illegal Argument Exception: Activation layer must be 1 or equal to the number of layers - 1 ");
            }

        } else {
            throw new Error("Illegal Argument Exception: Units: Units Must be of type array, not empty or one layer.");
        }
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

        let match = false
        for (let i = 0; i < this.initTypes.length; i++) {
            if (initType === this.initTypes[i]) {
                match = true;
                break;
            }
        }

        if (match === false) {
            throw new Error("Illegal Argument Exception: initType has has not been correctly defined");
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
                activationFunction: (this.activationFunctions.length === 1) ? this.activationFunctions[0] : this.activationFunctions[i - 1]
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

    mutate(rate) {

        // Need to go through each layer section - each weight & biases matrix

        for (let i = 0; i < this.layers.length; i++) {

            // update weights
            for (let j = 0; j < this.layers[i].weights.size().rows; j++) {
                for (let k = 0; k < this.layers[i].weights.size().cols; k++) {

                    if (Math.random() < rate) {
                        this.layers[i].weights.data[j][k] = Math.random() * 100;
                        //this.layers[i].weights.data[j][k] += Matrix2D.gaussian_distribution(0, 1, 1);
                    }

                }
            }

            // Mutate biases 
            for (let l = 0; l < this.layers[i].biases.size().rows; l++) {
                for (let p = 0; p < this.layers[i].biases.size().cols; p++) {

                    if (Math.random() < rate) {
                        this.layers[i].biases.data[l][p] = Math.random() * 100;
                        //this.layers[i].biases.data[l][p] += Matrix2D.gaussian_distribution(0, 1, 1);
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