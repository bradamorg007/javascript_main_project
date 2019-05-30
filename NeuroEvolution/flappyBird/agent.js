class Agent {

    constructor(empty) {

        // If you change the gravity and lift then the velocity limits also need changing
        this.yPos = height / 2;
        this.xPos = 50;
        this.gravity = 0.6;
        this.lift = -15;
        this.maxLim = 6;
        this.minLim = -15;
        this.velocity = 0;
        this.radius = 32;

        this.score = 0;

        let msLayerUnits = [6, 4, 2];
        let msActFunctions = ["RELU", "softmax"];

        this.brain = new NeuralNetwork(msLayerUnits, msActFunctions);

        if (empty === undefined) {
            this.brain.initLayers("he_normal");
        } else {
            this.brain.initLayers(); // init an empty array of elements
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

    update() {
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

        this.score++;
    }

    minMaxNormalise(x) {
        return (x - this.minLim) / (this.maxLim - this.minLim);
    }


}



if (typeof module !== 'undefined') {
    module.exports = Agent;
}