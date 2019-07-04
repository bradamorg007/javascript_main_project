from p5 import *
import numpy as np
from nn import NeuralNetwork


class Agent:

    def __init__(self, initBrain, initEmpty):

        self.yPos     = height/2
        self.xPos     = 50
        self.gravity  = 0.6
        self.lift     = -15
        self.maxLim   = 6
        self.minLim   = -15
        self.velocity = 0
        self.radius   = 25

        self.timeSamplesExperianced       = 0
        self.totalDistanceFromGapOverTime = 0

        self.fitness        = 0
        self.avgDistFromGap = 0

        # NEED TO ADD LOAD IN BRAIN FUNCTIONS BELOW:
        #
        #
        #
        #

        msLayeruUnits = [6, 4, 2]
        msActFunctions = ["relu", "softmax"]

        self.brain = NeuralNetwork(layer_units=msLayeruUnits, activation_func_list=msActFunctions)

        if initEmpty == False:
            self.brain.init_layers(init_type="he_normal")

        else:
            self.brain.init_layers(init_type="zeros")



    def show(self):
        fill(0, 100)
        stroke(0.5)
        ellipse(self.xPos, self.yPos, self.radius, self.radius)


    def think(self, closestBlock):

        inputs = []
        inputs.append(closestBlock.xPos / width)
        inputs.append(closestBlock.topStart / height)
        inputs.append(closestBlock.bottomStart / height)
        inputs.append((closestBlock.xPos - self.xPos) / width)
        inputs.append(self.yPos / height)
        inputs.append(self.minMaxNormalise(self.velocity))

        prediction = self.brain.feed_foward(inputs=inputs)

        if prediction[0] > prediction[1]:
            self.actionUp()


    def actionUp(self):
        self.velocity = self.velocity + self.lift


    def update(self, closestBlock):

        self.velocity = self.velocity + self.gravity
        self.velocity = self.velocity * 0.9
        self.yPos = self.yPos + self.velocity

        if self.velocity > self.maxLim:
            self.velocity = self.maxLim

        if self.velocity < self.minLim:
            self.velocity = self.minLim

        if self.yPos > height:
            self.yPos = height
            self.velocity = 0

        elif self.yPos < 0:
            self.yPos = 0
            self.velocity = 0

        # penalise agents for their distance on the y from the center of the gap of the blocks
        gap = closestBlock.bottomStart - closestBlock.topStart
        gapMid = closestBlock.topStart + np.round((gap / 2))
        agentDistanceFromGap = np.floor(np.abs(self.yPos - gapMid))

        self.totalDistanceFromGapOverTime = self.totalDistanceFromGapOverTime + agentDistanceFromGap
        self.timeSamplesExperianced = self.timeSamplesExperianced + 1

        self.fitness = self.fitness + 1


    def minMaxNormalise(self, x):
        return (x - self.minLim) / (self.maxLim - self.minLim)


    def computeFitness(self):
        # penalise agent based on average distance from gap

        impactFactor = 0.5 # scales the percentage of penalisation applied
        self.avgDistFromGap = np.floor(self.totalDistanceFromGapOverTime / self.timeSamplesExperianced)
        self.fitness = self.fitness - np.floor(impactFactor * self.avgDistFromGap)
        if self.fitness < 0:
            self.fitness = 0


if __name__ == "__main__":
    pass




