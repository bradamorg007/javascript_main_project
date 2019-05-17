 class Random {


     static integer(max) {
         // initialise matric with set of values 
         let output = [];
         for (let i = 0; i < this.rows; i++) {
             output[i] = [];
             for (let j = 0; j < this.cols; j++) {
                 output[i][j] = Math.floor(Math.random() * Math.floor(max));
             }
         }

         return output;
     }

     normal(rows, cols) {

         if (rows == 1 && cols == 1) {
             return gaussian(0, 1, 1);
         }
         let output = [];
         for (let i = 0; i < rows; i++) {
             output[i] = [];
             for (let j = 0; j < cols; j++) {
                 output[i][j] = gaussian(0, 1, 1);
             }
         }

         return output;

     }

     he_normal(rows, cols, mean, sigma) {
         if (rows == 1 && cols == 1) {
             return gaussian(0, 1, 1) * Math.sqrt(2 / cols);
         }
         let output = [];
         for (let i = 0; i < rows; i++) {
             output[i] = [];
             for (let j = 0; j < cols; j++) {
                 output[i][j] = gaussian(0, mean, sigma) * Math.sqrt(1 / (cols - 1));
             }
         }

         return output;
     }

     xavier_normal(rows, cols, mean, sigma) {
         if (rows == 1 && cols == 1) {
             return gaussian(0, mean, sigma) * Math.sqrt(2 / cols);
         }
         let output = [];
         for (let i = 0; i < rows; i++) {
             output[i] = [];
             for (let j = 0; j < cols; j++) {
                 output[i][j] = Random.gaussian(0, mean, sigma) * Math.sqrt(2 / (cols - 1));
             }
         }

         return output;
     }


     static gaussian(mean, sigma, samples) {

         if (!Number.isInteger(samples)) {
             console.error("Gaussian: Number of samples must be an int");
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