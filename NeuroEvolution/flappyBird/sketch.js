const TOTAL_POPULATION = 200;
const GA_ACTIVE = false;
const LOAD_MODEL = true;
let MODEL_LOAD_PATH = "resources/AgentBrain.json";

let activeAgents = [];;
let deadAgents = [];
let blocks = [];
let blockCounter = 0;
let generationCount = 0;

let slider;
let drawButton;
let saveButton;

let loadedBrain;

function preload() {

	if (LOAD_MODEL) {
		loadedBrain = loadJSON(MODEL_LOAD_PATH);
	}

}

function setup() {
	createCanvas(windowWidth * 0.7, windowHeight * 0.5);

	// Make the Interface Stuff
	slider = createSlider(1, 200, 1);
	drawButton = createCheckbox("draw sim", true);
	saveButton = createButton("Save Best Agent");
	saveButton.mousePressed(saveButtonPressed);

	loadAgents();


}

function loadAgents() {
	if (LOAD_MODEL) {

		//let brain = NeuralNetwork.deserialize(loadedModelJSON);
		//let brain = NeuralNetwork.deserialize(loadedBrain);
		let agent = new Agent(loadedBrain, true);
		activeAgents.push(agent);

	} else {
		for (let i = 0; i < TOTAL_POPULATION; i++) {
			activeAgents.push(new Agent(false, false));
		}
	}
}

function draw() {

	let cycles = slider.value();

	for (let c = 0; c < cycles; c++) {
		if (blockCounter % 75 === 0) {
			blocks.push(new block());
		}
		blockCounter++;


		for (let i = 0; i < blocks.length; i++) {

			blocks[i].update();

			if (blocks[i].offscreen()) {
				blocks.splice(i, 1);
				i--;
			}
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

			let agent = activeAgents[i];
			agent.think(clostestBlock);
			agent.update(clostestBlock);

			if (clostestBlock.hit(agent)) {
				// Now we penelise the agent based on their average distance from the center of the gaps
				// over the course of its entire life. this is to help distinguish between the performance
				// of agents that all crash into the same block, the ones that were closer to the gap, were
				// on a better track than the ones miles away from it.
				agent.computeFitness();
				deadAgents.push(agent);
				activeAgents.splice(i, 1);
				i--;
			}

		}

		// If everything is dead resart the game with new population.

		if (activeAgents.length === 0) {
			if (GA_ACTIVE) {
				activeAgents = GA.produceNextGeneration(deadAgents);
				generationCount++;
			} else {
				loadAgents();
			}
			resetGame();

			console.log(generationCount)
		}



	}

	if (drawButton.checked()) {
		// Draw Everything

		background(255);

		for (let i = 0; i < blocks.length; i++) {
			blocks[i].show();
		}

		for (let i = 0; i < activeAgents.length; i++) {
			activeAgents[i].show();
		}
	}



}

function resetGame() {
	deadAgents = [];
	blocks = [];
	blockCounter = 0;
}



function saveButtonPressed() {

	if (activeAgents.length > 0) {

		let a = activeAgents[0];
		a.computeFitness();
		let max = a.score;

		let index = 0;

		for (let i = 0; i < activeAgents.length; i++) {

			let currentAgent = activeAgents[i];
			currentAgent.computeFitness();

			if (currentAgent > max) {
				max = currentAgent.score;
				index = i;
			}

		}

		saveJSON(activeAgents[index].brain, 'AgentBrain.json');
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