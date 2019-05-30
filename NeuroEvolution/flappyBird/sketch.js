const TOTAL_POPULATION = 300;

let activeAgents = [];;
let deadAgents = [];
let blocks = [];

function setup() {
	createCanvas(windowWidth, windowHeight);

	for (let i = 0; i < TOTAL_POPULATION; i++) {
		activeAgents.push(new Agent());
	}
	blocks.push(new block());
}


function draw() {

	for (let i = 0; i < blocks.length; i++) {

		blocks[i].update();

		if (blocks[i].offscreen()) {
			blocks.splice(i, 1);
			i--;
		}
	}

	if (frameCount % 100 === 0) {
		blocks.push(new block());
	}

	// All agents will have the same xPos 
	let globalXPos = activeAgents[0].xPos;

	// Find the closestBloct. because the agets all have the same xPos then 
	// it means the closest block for one will be the closest block for all

	let clostestBlock = null;

	for (let i = 0; i < blocks.length; i++) {
		if (globalXPos < (blocks[i].xPos + blocks[i].width)) {
			clostestBlock = blocks[i];
			break;
		}
	}

	// Go through each agent check if they have hit something and if they have 
	// add them to the deadAgents list. 

	for (let i = 0; i < activeAgents.length; i++) {

		activeAgents[i].think(clostestBlock);
		activeAgents[i].update();

		if (clostestBlock.hit(activeAgents[i])) {
			deadAgents.push(activeAgents[i]);
			activeAgents.splice(i, 1);
			i--;
		}

	}


	// Draw Everything

	background(255);

	for (let i = 0; i < blocks.length; i++) {
		blocks[i].show();
	}

	for (let i = 0; i < activeAgents.length; i++) {
		activeAgents[i].show();
	}

	// If everything is dead resart the game with new population.

	if (activeAgents.length === 0) {
		activeAgents = GA.produceNextGeneration(deadAgents);
		deadAgents = [];
		//blocks = [];
	}

}


// function resetGame() {
// 	deadAgents = [];
// 	activeAgents = [];
// 	for (let i = 0; i < TOTAL_POPULATION; i++) {
// 		activeAgents.push(new Agent());
// 	}
// }


// p5 if you define a function in the sketch it will get executed
// The keyPressed() function is called once every time a key is pressed. 
//The keyCode for the key that was pressed is stored in the keyCode variable. 
// function keyPressed() {

// 	if (key === ' ') {
// 		agent.actionUp();
// 	}
// }