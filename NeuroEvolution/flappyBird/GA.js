class GA {

    static mutateRate = 0.1;
    static selectionType = "roullete"; // can add tornoment
    static crossOverRate = 0.5; // Chance that child gene will be fromparent A or B

    static produceNextGeneration(population) {

        // To produce the next generation several steps are required

        // NOTE 
        // GA Could select Parent A and B from the same neural network if its very high fitness
        // So the GA may favor one particular network, but mutation can still happen to keep things 
        // changing

        // 1.) sum the fitness and add exponetial curvature to fitness
        // We add exponential curvature to greater distinctions between the performance of agents
        // e.g A fitness = 20, B fitness = 19 here the fitness difference is very small only one.
        // both A and B will have a very simular probability of been selected even tho A is better
        // but 20^2 = 400 and 19^2 = 361 this creates a muc bigger difference between their performances
        // now A is much more likely to be selected than B

        // Order Population from largest fitness to smallest. larger fitness are more likely to be selected
        // so we might aswell iterate through them first, allows use to break out of the list
        let newPopulation = [];
        let fitnessSum = 0;

        for (let i = 0; i < population.length; i++) {
            population[i].score = Math.pow(population[i].score, 2);
            fitnessSum += population[i].score;
        }


        // 2.) Proportional fitness probabilities Normalise the agent scores now that we have the sum. 
        for (let i = 0; i < population.length; i++) {
            population[i].score = population[i].score / fitnessSum;
        }

        // 3.) now I need to create a new population of children

        // i - throw two darts to choose Parent A and B
        for (let i = 0; i < population.length; i++) {

            let parentA = null;
            let parentB = null;

            let r1 = Math.random();
            let r2 = Math.random();

            let index1 = 0;
            let index2 = 0;

            let done1 = false;
            let done2 = false;
            let doneOverall = false;
            while (doneOverall === false) {

                r1 -= population[index1].score;
                r2 -= population[index2].score;

                if (r1 < 0) {
                    done1 = true;

                    if (parentA === null) {
                        parentA = population[index1];

                    } else {
                        parentB = population[index1];
                    }
                }

                if (r2 < 0) {
                    done2 = true;

                    if (parentA === null) {
                        parentA = population[index2];

                    } else {
                        parentB = population[index2];

                    }
                }

                if (done1 === true && done2 === true) {
                    doneOverall = true;

                } else {

                    if (done1 === false) {
                        index1++;
                    }

                    if (done2 === false) {
                        index2++;
                    }

                }
            }


            newPopulation.push(GA.reproduceAndMutate(parentA, parentB));

        }

        return newPopulation;

    }

    static reproduceAndMutate(parentA, parentB) {

        // Now go through Parents parmaters and exchange gentic info
        //  Also mutate select gene within the same loop
        // no need having a seperte mutate function that loops through paramter matrices again

        // Loops can use child dimensions as all networks have fixed same topologies in this

        let child = new Agent(true); // init an empty child

        for (let i = 0; i < child.brain.layers.length; i++) {

            // update weights
            for (let j = 0; j < child.brain.layers[i].weights.size().rows; j++) {
                for (let k = 0; k < child.brain.layers[i].weights.size().cols; k++) {

                    if (Math.random() < GA.crossOverRate) {
                        // use ParentA gene
                        child.brain.layers[i].weights.data[j][k] = parentA.brain.layers[i].weights.data[j][k];

                        if (Math.random() < GA.mutateRate) {
                            child.brain.layers[i].weights.data[j][k] += Matrix2D.gaussian_distribution(0, 1, 1);
                        }

                    } else {
                        // Parent B gene
                        child.brain.layers[i].weights.data[j][k] = parentB.brain.layers[i].weights.data[j][k];

                        if (Math.random() < GA.mutateRate) {
                            child.brain.layers[i].weights.data[j][k] += Matrix2D.gaussian_distribution(0, 1, 1);
                        }

                    }

                }
            }

            // Mutate biases 
            for (let l = 0; l < child.brain.layers[i].biases.size().rows; l++) {
                for (let p = 0; p < child.brain.layers[i].biases.size().cols; p++) {

                    if (Math.random() < GA.mutateRate) {
                        // use ParentA gene
                        child.brain.layers[i].biases.data[l][p] = parentA.brain.layers[i].biases.data[l][p];

                        if (Math.random() < GA.mutateRate) {
                            child.brain.layers[i].biases.data[l][p] += Matrix2D.gaussian_distribution(0, 1, 1);
                        }

                    } else {
                        // Parent B gene
                        child.brain.layers[i].biases.data[l][p] = parentB.brain.layers[i].biases.data[l][p];

                        if (Math.random() < GA.mutateRate) {
                            child.brain.layers[i].biases.data[l][p] += Matrix2D.gaussian_distribution(0, 1, 1);
                        }

                    }
                }
            }

        }

        return child;

    }

}