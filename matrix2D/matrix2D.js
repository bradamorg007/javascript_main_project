class Matrix2D {

    constructor(rows, cols) {

        this.rows = rows;
        this.cols = cols;
        this.data = [];

    }


    randInit(max) {
        // initialise matric with set of values 

        for (let i = 0; i < this.rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = Math.floor(Math.random() * Math.floor(max));
            }
        }


    }



    static gaussian(mean, sigma, samples) {

        if (!Number.isInteger(samples)) {
            console.error("Gaussian: Number of samples must be an int");
        }
        // loop over the number of samples needed

        let two_pi = Math.PI * 2;
        let output = [];

        for (let i = 1; i < samples / 2; i++) {

            // sample two points from uniform distribution between 0-1
            let u1 = Math.random();
            let u2 = Math.random();


            if (u1 == 0) {
                let z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(two_pi * u2);
                output[i] = (z * sigma) + mean;
            } else {
                output[i] = 0;
            }

            if (u2 == 0) {
                z = Math.sqrt(-2 * Math.log(u1)) * Math.sin(two_pi * u2);
                output[i + 1] = (z * sigma) + mean;
            } else {
                output[i + 1] = 0;
            }
        }

        return output;

    }


    print() {
        console.table(this.data);
    }




}