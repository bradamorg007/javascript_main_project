class Agent {

    constructor(initBrain, initEmpty) {

        // If you change the gravity and lift then the velocity limits also need changing. This is max & minLim
        this.yPos = height / 2;
        this.xPos = 50;
        this.gravity = 0.6;
        this.lift = -15;
        this.maxLim = 6;
        this.minLim = -15;
        this.velocity = 0;
        this.radius = 25;

        this.timeSamplesExperianced = 0;
        this.totalDistanceFromGapOverTime = 0;

        this.score = 0;
        this.fitness = 0;
        this.avgDistFromGap = 0;

        if (initBrain.CLASS_TAG !== undefined || initBrain === true) {
            if (initBrain.CLASS_TAG === "nn.js") {

                this.brain = NeuralNetwork.deserialize(initBrain);
            }

        } else {

            let msLayerUnits = [6, 4, 2];
            let msActFunctions = ["RELU", "softmax"];

            this.brain = new NeuralNetwork(msLayerUnits, msActFunctions);

            if (initEmpty === false) {
                this.brain.initLayers("he_normal");
            } else {
                this.brain.initLayers(); // init an empty array of elements
            }
        }

    }

    show() {
        fill(0, 100);
        stroke(0.5);
        ellipse(this.xPos, this.yPos, this.radius, this.radius);
    }

    think(clostestBlock) {

        // If the screen is too small not enough blocks will spawn meaning
        // that clostestBlock will stay null. So just increase screen size this error will go away
        // this should be a consideration when using pixel input.

        let inputs = [];
        inputs[0] = clostestBlock.xPos / width;
        inputs[1] = clostestBlock.topStart / height;
        inputs[2] = clostestBlock.bottomStart / height;
        inputs[3] = (clostestBlock.xPos - this.xPos) / width; // This the distance from agent to the next block
        inputs[4] = this.yPos / height;
        inputs[5] = this.minMaxNormalise(this.velocity); // Need to Normalise this some how?? Use an activation function like sigmoid
        // it can squash the numbers but if the velocity is too big it will ceiling???? 

        let prediction = this.brain.feedFoward(inputs);

        // console.log(prediction.data[0] + "  " + prediction.data[1]);
        if (prediction.data[0] > prediction.data[1]) {
            this.actionUp();
        }

        // determine which pipe is closest
        // 

    }

    actionUp() {
        this.velocity += this.lift;
    }

    update(clostestBlock) {
        //  The velcoity cumlatively increases with respect to the amount of gravity
        // Update the yPosition with the velocity. y = the top is 0 and the bottom of the screen is the height 
        // of the window. 

        this.velocity += this.gravity;
        this.velocity *= 0.9;
        this.yPos += this.velocity;

        if (this.velocity > this.maxLim) {
            this.velocity = this.maxLim;
        }

        if (this.velocity < this.minLim) {
            this.velocity = this.minLim;
        }
        // Stop the agent from falling past the screen limits
        if (this.yPos > height) {
            this.yPos = height;
            this.velocity = 0;

        } else if (this.yPos < 0) {
            this.yPos = 0;
            this.velocity = 0;

        }

        // penelise agents for their distance on the y from the center of the gap of the obsticles
        let gap = clostestBlock.bottomStart - clostestBlock.topStart;
        let gapMid = clostestBlock.topStart + Math.round((gap / 2));
        let agentDistanceFromGap = Math.floor(Math.abs(this.yPos - gapMid));

        this.totalDistanceFromGapOverTime += agentDistanceFromGap;
        this.timeSamplesExperianced++;

        this.score++;
    }

    minMaxNormalise(x) {
        return (x - this.minLim) / (this.maxLim - this.minLim);
    }

    computeFitness() {

        // penalise agent based on avg distance from gap

        let impactFactor = 0.5; // adjusts the percentage of penalisation applied
        this.avgDistFromGap = Math.floor(this.totalDistanceFromGapOverTime / this.timeSamplesExperianced);
        this.score -= Math.floor(impactFactor * this.avgDistFromGap);
        if (this.score < 0) {
            this.score = 0;
        }
    }


}



if (typeof module !== 'undefined') {
    module.exports = Agent;
}