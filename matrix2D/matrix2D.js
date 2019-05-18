// MATHS OPERATION FUNCTIONS
// scalars
// add, multply, subtract
// Ygit commi
// element wise 
// add, multiply, substract *
// need a check function to see if the the dimesnions match up *

// dot product multplication between matrices and vectors *
// need a check function to make sure the dimensions match up * 

// GPU implementations of the above mathematical operations

// OTHER FUNCTIONS

// get size function *
// flatten function *
// transpose function *
// Print matrices *
// init matrix - use random generation using int, normal distribution,*
// gaussian, xaviar or he_normal distributions or truncated normal *
// Or user specified data input 

function isNumeric(n) {
    return !isNaN(parseFloat(n)) && isFinite(n);
}


function checkInputsForMatrixOperation(a, b) {

    let aIsScalar = false;
    let bIsScalar = false;

    if (!(a instanceof Matrix2D) && isNumeric(a) && !(a instanceof Array)) {
        aIsScalar = true;
    }

    if (!(b instanceof Matrix2D) && isNumeric(b) && !(b instanceof Array)) {
        bIsScalar = true;
    }

    if (aIsScalar && bIsScalar) {
        return a * b;
    }

    return {
        aIsScalar: aIsScalar,
        bIsScalar: bIsScalar
    };

}

class Matrix2D {

    constructor(rows, cols, input) {

        if (rows == 0 || cols == 0) {
            throw new Error("Illegal Argument Exception: rows or cols can not be set to 0");
        }

        this.rows = rows;
        this.cols = cols;
        this.previousRows = 0;
        this.previousCols = 0;
        this.data = [];

        if (input == undefined) {
            this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
        } else {
            if ((rows * cols) !== input.length) {
                throw new Error("Illegal Argument Exception: The dimensions of the" +
                    "input data must be compatible with the dimensions of the rows and cols");
            }

            let count = 0;
            for (let i = 0; i < rows; i++) {
                this.data[i] = [];
                for (let j = 0; j < cols; j++) {
                    this.data[i][j] = input[count];
                    count++;
                }
            }
        }


    }



    print(...data) {
        console.table(this.data);

        if (data.length > 0) {
            for (let i = 0; i < data.length; i++) {
                console.table(data[i].data);
            }
        }
    }

    // MATH OPERATIONS =====================================================================

    static dotProduct(a, b) {

        // Cols of a == rows of b 
        let checkOutput = checkInputsForMatrixOperation(a, b);

        let aIsScalar = checkOutput.aIsScalar;
        let bIsScalar = checkOutput.bIsScalar;

        if (!aIsScalar && !bIsScalar) {
            if (a.size().cols === b.size().rows) {

                // This will make a new Matrix equal to the size of the matrix product 
                // given sizes of a and b. then the map function will iterate over 
                // all the element positions of the matrix and apply the below function to each element
                // the rows of the new matrix product will be equal to the rows of a. 
                // the columns of the new matrix product will be equal to the columns of b
                // The columns a must == rows of b, thus k can be used to index both the 
                // values in the columsn of a with the values in the rows of b....AHHHH! BRAIN 
                return new Matrix2D(a.size().rows, b.size().cols)
                    .map((e, i, j) => {
                        let sum = 0;
                        for (let k = 0; k < a.size().cols; k++) {
                            sum += a.data[i][k] * b.data[k][j];
                        }
                        return sum;
                    });

            } else {
                throw new Error("Illegal Argument Exception: dot product requires the cols of a == rows of b ");
            }
        } else {
            throw new Error("Illegal Argument Exception: dot product requires both inputs be Matrix2D Objects");
        }

    }


    static subtract(a, b) {

        let checkOutput = checkInputsForMatrixOperation(a, b);

        let aIsScalar = checkOutput.aIsScalar;
        let bIsScalar = checkOutput.bIsScalar;

        if (aIsScalar && !bIsScalar) {
            return new Matrix2D(b.size().rows, b.size().cols)
                .map((_, i, j) => b.data[i][j] - a);

        } else if (!aIsScalar && bIsScalar) {
            return new Matrix2D(a.size().rows, a.size().cols)
                .map((_, i, j) => a.data[i][j] - b);

        } else if (!aIsScalar && !bIsScalar) {

            if (a.size().rows !== b.size().rows || a.size().cols !== b.size().cols) {
                throw new Error("Illegal Argument Exception: Element Wise Operations require input matrices to have the same dimensions");
            }

            return new Matrix2D(a.size().rows, a.size().cols)
                .map((_, i, j) => a.data[i][j] - b.data[i][j]);
        }
    }


    static add(a, b) {

        let checkOutput = checkInputsForMatrixOperation(a, b);

        let aIsScalar = checkOutput.aIsScalar;
        let bIsScalar = checkOutput.bIsScalar;

        if (aIsScalar && !bIsScalar) {
            return new Matrix2D(b.size().rows, b.size().cols)
                .map((_, i, j) => b.data[i][j] + a);

        } else if (!aIsScalar && bIsScalar) {
            return new Matrix2D(a.size().rows, a.size().cols)
                .map((_, i, j) => a.data[i][j] + b);

        } else if (!aIsScalar && !bIsScalar) {

            if (a.size().rows !== b.size().rows || a.size().cols !== b.size().cols) {
                throw new Error("Illegal Argument Exception: Element Wise Operations require input matrices to have the same dimensions");
            }

            return new Matrix2D(a.size().rows, a.size().cols)
                .map((_, i, j) => a.data[i][j] + b.data[i][j]);
        }
    }


    static multiply(a, b) {

        let checkOutput = checkInputsForMatrixOperation(a, b);

        let aIsScalar = checkOutput.aIsScalar;
        let bIsScalar = checkOutput.bIsScalar;

        if (aIsScalar && !bIsScalar) {
            return new Matrix2D(b.size().rows, b.size().cols)
                .map((_, i, j) => b.data[i][j] * a);

        } else if (!aIsScalar && bIsScalar) {
            return new Matrix2D(a.size().rows, a.size().cols)
                .map((_, i, j) => a.data[i][j] * b);

        } else if (!aIsScalar && !bIsScalar) {

            if (a.size().rows !== b.size().rows || a.size().cols !== b.size().cols) {
                throw new Error("Illegal Argument Exception: Element Wise Operations require input matrices to have the same dimensions");
            }

            return new Matrix2D(a.size().rows, a.size().cols)
                .map((_, i, j) => a.data[i][j] * b.data[i][j]);
        }
    }


    // OTHER FUNCTIONS =====================================================================================

    map(func) {

        for (let i = 0; i < this.size().rows; i++) {
            for (let j = 0; j < this.size().cols; j++) {
                let val = this.data[i][j];
                this.data[i][j] = func(val, i, j);
            }
        }

        return this;
    }

    transpose() {
        let m = new Matrix2D(this.size().cols, this.size().rows)
            .map((_, i, j) => this.data[j][i]);

        this.copy(m);
    }


    copy(matrix) {

        if (matrix instanceof Matrix2D) {
            this.rows = matrix.size().rows;
            this.cols = matrix.size().cols;
            this.previousRows = matrix.previousRows;
            this.previousCols = matrix.previousCols;
            this.data = matrix.data;
        } else {
            throw new Error("Illegal Argument Exception:" +
                "Copy requires a Matrix2D object be passed as an argument");
        }
    }


    flatten(flatternType) {

        if (flatternType == undefined) {
            throw new Error("Illegal Argument Exception: flattenType undefined");
        }

        if (this.data.length == 0) {
            throw new Error("Illegal Argument Exception: NOTHING TO FLATTEN Object data is empty");
        }

        if (this.size().rows == 1 || this.size().cols == 1) {
            return;
        }
        // if C style flatten by rows. so first go through all the values
        // for row one then row 2 and so on.
        // if F style (fortran) then go by collumns first
        let size = this.size();
        let output = [...Array(size.rows * size.cols)];
        this.previousRows = this.rows;
        this.previousCols = this.cols;
        this.rows = size.rows * size.cols;
        this.cols = 1;

        let count = 0;
        switch (flatternType) {

            case "C":

                for (let i = 0; i < size.cols; i++) {
                    for (let j = 0; j < size.rows; j++) {
                        output[count] = this.data[j][i];
                        count++;
                    }
                }

                this.data = output;
                break;

            case "F":
                for (let i = 0; i < size.rows; i++) {
                    for (let j = 0; j < size.cols; j++) {
                        output[count] = this.data[i][j];
                        count++;
                    }
                }

                this.data = output;
                break;
        }


    }


    unFlatten(flatternType) {

        if (flatternType == undefined) {
            throw new Error("Illegal Argument Exception: flattenType undefined");
        }

        if (this.data.length == 0) {
            throw new Error("Illegal Argument Exception: NOTHING TO FLATTEN Object data is empty");
        }

        if (this.previousRows == 0 || this.previousCols == 0) {
            throw new Error("Illegal Argument Exception: NOTHING TO UNFLATTEN data is already a 2D matrix");
        }

        if (this.size().rows > 1 && this.size().cols > 1) {
            return;
        }
        // if C style flatten by rows. so first go through all the values
        // for row one then row 2 and so on.
        // if F style (fortran) then go by collumns first
        let size = this.size();
        let output = Array(this.previousRows).fill(null).map(() => Array(this.previousCols).fill(null));
        this.rows = this.previousRows;
        this.cols = this.previousCols;


        let count = 0;
        switch (flatternType) {

            case "C":
                for (let i = 0; i < this.previousCols; i++) {
                    for (let j = 0; j < this.previousRows; j++) {
                        output[j][i] = this.data[count];
                        count++;
                    }
                }

                this.data = output;
                break;

            case "F":
                for (let i = 0; i < this.previousRows; i++) {
                    for (let j = 0; j < this.previousCols; j++) {
                        output[i][j] = this.data[count];
                        count++;
                    }
                }

                this.data = output;
                break;
        }

        this.previousRows = 0;
        this.previousCols = 0;

    }


    toArray() {

        let output = [];
        for (let i = 0; i < this.size.rows; i++) {
            for (let j = 0; j < this.size.cols; j++) {
                output.push(this.data[i][j]);
            }
        }
        return output;
    }

    // GETTERS AND SETTERS

    size() {
        return {
            rows: this.rows,
            cols: this.cols
        };
    }




    // INIT MATRIX ELEMENTS FUNCTIONS =================================================

    integer(max) {
        // initialise matric with set of values 
        let size = this.size();
        for (let i = 0; i < size.rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < size.cols; j++) {
                this.data[i][j] = Math.floor(Math.random() * Math.floor(max));
            }
        }
    }

    normal() {

        let size = this.size();

        if (size.rows == 1 && size.cols == 1) {
            return gaussian_distribution(0, 1, 1);
        }
        for (let i = 0; i < size.rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < size.cols; j++) {
                this.data[i][j] = Matrix2D.gaussian_distribution(0, 1, 1);
            }
        }
    }

    he_normal() {

        // For RELU 
        let mean = 0;
        let sigma = 1;
        let samples = 1;
        let cuttoff = 2;
        let size = this.size();

        if (size.rows == 1 && size.cols == 1) {
            let draw = Matrix2D.truncated_gaussian_distribution(mean, sigma, samples, cuttoff);
            return draw * Math.sqrt(2 / (size.cols));
        }
        for (let i = 0; i < size.rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < size.cols; j++) {
                let draw = Matrix2D.truncated_gaussian_distribution(mean, sigma, samples, cuttoff);
                this.data[i][j] = draw * Math.sqrt(2 / (size.cols));
            }
        }
    }

    // Make truncated normal function this where all values greater than 2stds from the
    // the means of 0 will be disgarded and redrawn
    // xavier and he normal will draw from a truncated normal then times its value by sqr(2 or 1 / cols -1)


    xavier_normal() {

        // For Tanh
        // Weight Matrix = rows = neurons - cols = number of weights connected to the neuron
        // the weight matrix shape is defined. W = [rows = number of neurons in current Layer, cols number of neurons in previous layer so the layer before the current layer];

        // First create the normally distributed points;
        let mean = 0;
        let sigma = 1;
        let samples = 1;
        let cuttoff = 2;
        let size = this.size();

        if (size.rows == 1 && size.cols == 1) {
            return this.truncated_normal(mean, sigma, samples, cuttoff) * Math.sqrt(1 / (size.cols));
        }
        for (let i = 0; i < size.rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < size.cols; j++) {
                let draw = Matrix2D.truncated_gaussian_distribution(mean, sigma, samples, cuttoff);
                this.data[i][j] = draw * Math.sqrt(1 / (size.cols));
            }
        }

    }

    truncated_normal() {

        let mean = 0;
        let sigma = 1;
        let samples = 1;
        let cuttoff = 2;
        let size = this.size();

        if (size.rows == 1 && size.cols == 1) {
            return Matrix2D.truncated_gaussian_distribution(mean, sigma, samples, cuttoff);
        }
        for (let i = 0; i < size.rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < size.cols; j++) {
                this.data[i][j] = Matrix2D.truncated_gaussian_distribution(mean, sigma, samples, cuttoff);
            }
        }

    }

    static truncated_gaussian_distribution(mean, sigma, samples, cuttOff) {

        if (!Number.isInteger(samples)) {
            throw new Error("Gaussian: Number of samples must be an int");
        }

        if (samples === 0) {
            throw new Error("Illegal Argument Exception: Sample number must be greater than 0");
        }

        let output = [...Array(samples)];

        for (let i = 0; i < samples; i++) {
            let done = false;
            while (done == false) {
                let draw = Matrix2D.gaussian_distribution(mean, sigma, 1);
                if (draw <= (mean + cuttOff) && draw >= (mean - cuttOff)) {

                    output[i] = draw;

                    if (samples === 1) {
                        return draw;
                    }

                    done = true;
                }
            }
        }

        return output;

    }

    static gaussian_distribution(mean, sigma, samples) {

        if (!Number.isInteger(samples)) {
            throw new Error("Gaussian: Number of samples must be an int");
        }
        // loop over the number of samples needed

        let two_pi = Math.PI * 2;
        let output = [];

        for (let i = 0; i < samples / 2; i++) {

            // sample two points from uniform distribution between 0-1
            let u1 = Math.random();
            let u2 = Math.random();

            let z = 0;
            if (u1 != 0) {
                z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(two_pi * u2);
                output[i] = (z * sigma) + mean;
            } else {
                output[i] = 0;
            }

            if (samples == 1) {
                return output[0];
            }

            if (u2 != 0) {
                z = Math.sqrt(-2 * Math.log(u1)) * Math.sin(two_pi * u2);
                output[i + 1] = (z * sigma) + mean;
            } else {
                output[i + 1] = 0;
            }

            i++
        }
        return output;
    }



}

if (typeof module !== 'undefined') {
    module.exports = Matrix2D;
}