class block {

    constructor() {

        this.width = 100;
        this.speed = 5;
        this.xPos = width;
        let gapSize = 70;

        // Math.floor(Math.random() * (max - min + 1)) + min     ceil(max), floor(min)

        let topStart = Math.random() * (height - 50);
        let bottomStart = Math.floor(Math.random() * (height - ((topStart + gapSize) + 1))) + (topStart + gapSize);

        this.topStart = topStart;
        this.bottomStart = bottomStart;

    }

    show() {

        fill(0);
        rect(this.xPos, 0, this.width, this.topStart);
        rect(this.xPos, this.bottomStart, this.width, height);
    }

    update() {
        this.xPos -= this.speed;
    }

    hit(agent) {

        if (agent.yPos - (agent.radius * 0.5) < this.topStart || agent.yPos + (agent.radius * 0.5) > this.bottomStart) {
            if (agent.xPos > this.xPos && agent.xPos < this.xPos + this.width) {
                return true;
            }
        }

        return false;
    }

    offscreen() {
        if (this.xPos < -this.width) {
            return true;
        } else {
            return false;
        }
    }


}

if (typeof module !== 'undefined') {
    module.exports = block;
}