import pygame
import numpy as np
from nn import NeuralNetwork


class Agent(pygame.sprite.Sprite):

    def __init__(self, initBrain, initEmpty, screen_width, screen_height, name='agent'):
        # Call the parent's constructor
        super().__init__()

        self.gravity  = 0.6
        self.lift     = -2
        self.maxLim   = 6
        self.minLim   = -15
        self.velocity = 0
        self.radius   = 2
        self.color = (0, 0, 0, 50)

        self.name = name
        self.image = pygame.Surface([self.radius, self.radius], pygame.SRCALPHA)
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.x = 10
        self.rect.y = screen_height/2

        self.timeSamplesExperianced       = 0
        self.totalDistanceFromGapOverTime = 0

        self.fitness        = 0
        self.avgDistFromGap = 0

        # NEED TO ADD LOAD IN BRAIN FUNCTIONS BELOW:
        #
        #
        #
        #

        msLayeruUnits = [9, 4, 2]
        msActFunctions = ["relu", "softmax"]

        self.brain = NeuralNetwork(layer_units=msLayeruUnits, activation_func_list=msActFunctions)

        if initEmpty == False:
            self.brain.init_layers(init_type="he_normal")

        else:
            self.brain.init_layers(init_type="zeros")



    def show(self):
        pass


    def think(self, closestBlock, screen_width, screen_height):

        inputs = []
        # input about on coming block object
        inputs.append(closestBlock.xPos / screen_width)
        inputs.append(closestBlock.top_block.rect.bottom / screen_height)
        inputs.append(closestBlock.bottom_block.rect.top / screen_height)
        inputs.append((closestBlock.xPos - self.rect.right) / screen_width)

        # input about agents current position
        inputs.append(self.rect.bottom / screen_height)
        inputs.append(self.rect.top / screen_height)
        inputs.append(self.rect.right / screen_height)
        inputs.append(self.rect.left / screen_height)
        inputs.append(self.minMaxNormalise(self.velocity))

        inputs = np.array([inputs])

        prediction = self.brain.feed_foward(inputs=inputs)

        if prediction[0] > prediction[1]:
            self.actionUp()


    def actionUp(self):
        self.velocity += self.lift


    def update(self, closestBlock, screen_height):

        self.velocity += self.gravity
        self.velocity *= 0.9
        self.rect.y += self.velocity

        if self.velocity > self.maxLim:
            self.velocity = self.maxLim

        if self.velocity < self.minLim:
            self.velocity = self.minLim

        if self.rect.bottom > screen_height:
            self.rect.bottom = screen_height
            self.velocity = 0

        elif self.rect.top < 5:
            self.rect.top = 0
            self.velocity = 0

        # penalise agents for their distance on the y from the center of the gap of the blocks
        gap = closestBlock.bottom_block.rect.top - closestBlock.top_block.rect.bottom
        gapMid = closestBlock.top_block.rect.bottom + np.round((gap / 2))
        agentDistanceFromGap = np.floor(np.abs(self.rect.midright[1] - gapMid))

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




