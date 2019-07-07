import math
import numpy as np
from agent import Agent
from Random import Random

class GA:

    mutateRate = 0.1
    selectionType = "roullete"
    crossOverRate = 0.5

    @staticmethod
    def produceNextGeneration(population, screen_width, screen_height):

        # // To produce the next generation several steps are required
        #
        # // NOTE
        # // GA Could select Parent A and B from the same neural network if its very high fitness
        # // So the GA may favor one particular network, but mutation can still happen to keep things
        # // changing
        #
        # // 1.) sum the fitness and add exponetial curvature to fitness
        # // We add exponential curvature to greater distinctions between the performance of agents
        # // e.g A fitness = 20, B fitness = 19 here the fitness difference is very small only one.
        # // both A and B will have a very simular probability of been selected even tho A is better
        # // but 20^2 = 400 and 19^2 = 361 this creates a muc bigger difference between their performances
        # // now A is much more likely to be selected than B
        #
        # // Order Population from largest fitness to smallest. larger fitness are more likely to be selected
        # // so we might aswell iterate through them first, allows use to break out of the list

        newPopulation = []
        fitnessSum = 0

        # FITNESS FUNCTION: uses power fo non_linear Fitness
        for i in range(len(population)):
            population[i].fitness = math.pow(population[i].fitness, 2)
            population[i].computeFitness()
            fitnessSum = fitnessSum + population[i].fitness

        # // // 2.) Proportional fitness probabilities Normalise the agent fitnesss now that we have the sum.
        # // for (let i = 0; i < population.length; i++) {
        # //     population[i].fitness = population[i].fitness / fitnessSum;
        # // }
        #
        # // 3.) now I need to create a new population of children

        # i throw two darts to choose Parent A and B
        for i in range(len(population)):

            parentA = GA.selectParent(population, fitnessSum)
            parentB = GA.selectParent(population, fitnessSum)
            newPopulation.append(GA.reproduceAndMutate(parentA, parentB, screen_width, screen_height))

        return newPopulation

    @staticmethod
    def selectParent(population, fitnessSum):

        index = 0
        r = np.round(np.random.rand() * fitnessSum)

        while r > 0:
            r = r - population[index].fitness

            if r > 0:
                index = index + 1

        parent = population[index]

        if parent == None:
            raise ValueError("ERROR GA: Parent in select parent method is undefined this is due to the indexing")

        return parent

    @staticmethod
    def reproduceAndMutate(parentA, parentB, screen_width, screen_height):
        # // Now go through Parents parmaters and exchange gentic info
        # //  Also mutate select gene within the same loop
        # // no need having a seperte mutate function that loops through paramter matrices again
        #
        # // Loops can use child dimensions as all networks have fixed same topologies in this

        child = Agent(initBrain=False, initEmpty=True, screen_width=screen_width, screen_height=screen_height )


        for i in range(len(child.brain.layers)):

            rowsW = child.brain.layers[i]['weights'].shape[0]
            colsW = child.brain.layers[i]['weights'].shape[1]

            for j in range(rowsW):
                for k in range(colsW):

                    if np.random.rand() < GA.crossOverRate:
                        # Use Parent A gene
                        child.brain.layers[i]['weights'][j][k] = parentA.brain.layers[i]['weights'][j][k]

                    else:
                        child.brain.layers[i]['weights'][j][k] = parentB.brain.layers[i]['weights'][j][k]


                    if np.random.rand() < GA.mutateRate:
                        child.brain.layers[i]['weights'][j][k] = child.brain.layers[i]['weights'][j][k] + Random.gaussian_distribution(mean=0, sigma=0,samples=1)

            # Reproduce and Mutate Baiases
            rowsB = child.brain.layers[i]['biases'].shape[0]
            colsB = child.brain.layers[i]['biases'].shape[1]

            for j in range(rowsB):
                for k in range(colsB):

                    if np.random.rand() < GA.crossOverRate:
                        # Use Parent A gene
                        child.brain.layers[i]['biases'][j][k] = parentA.brain.layers[i]['biases'][j][k]

                    else:
                        child.brain.layers[i]['biases'][j][k] = parentB.brain.layers[i]['biases'][j][k]

                    if np.random.rand() < GA.mutateRate:
                        child.brain.layers[i]['biases'][j][k] = child.brain.layers[i]['biases'][j][
                                                                     k] + Random.gaussian_distribution(mean=0, sigma=0,
                                                                                                       samples=1)
        return child



if __name__ == "__main__":

    parentA = Agent(initBrain=False, initEmpty=False, screen_width=800, screen_height=300)
    parentB = Agent(initBrain=False, initEmpty=False, screen_width=800, screen_height=300)
    child = GA.reproduceAndMutate(parentA, parentB, screen_width=800, screen_height=300)

