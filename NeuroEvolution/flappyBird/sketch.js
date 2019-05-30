let agent;
let blocks = [];

function setup() {
	createCanvas(windowWidth, windowHeight);
	agent = new Agent();
	blocks.push(new block());
}


function draw() {
	background(255);


	if (frameCount % 100 === 0) {
		blocks.push(new block());
	}

	for (let i = 0; i < blocks.length; i++) {

		if (blocks[i].hit(agent)) {
			//console.log("HIT!!!");
			//resetGame();
			//break;

		}

		blocks[i].update();
		blocks[i].show();

		if (blocks[i].offscreen()) {
			blocks.splice(i, 1);
			i--;
		}


	}

	agent.think(blocks);
	agent.update();
	agent.show();

}


function resetGame() {
	// For Now this just resets the agent. But for the genetic algo we need it to kill the agent.
	// agent.yPos = height / 2;
	// blocks = [];
}


// p5 if you define a function in the sketch it will get executed
// The keyPressed() function is called once every time a key is pressed. 
//The keyCode for the key that was pressed is stored in the keyCode variable. 
// function keyPressed() {

// 	if (key === ' ') {
// 		agent.actionUp();
// 	}
// }