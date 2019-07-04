import numpy as np
from p5 import *
from agent import Agent
from block import Block
from GA import GA

class Simulation:

    def __init__(self):

        self.TOTAL_POPULATION = 400
        self.GA_ACTIVE = True
        self.LOAD_MODEL = False
        self.MODEL_LOAD_PATH = ""
        self.CYCLES = 1

        self.activeAgents = []
        self.deadAgents = []
        self.blocks = []
        self.blockCounter = 0
        self.generationCount = 0

        self.slider = None
        self.drawButton = None
        self.saveButton = None

        self.loadedBrain = None



def setup():

    size(800, 300)
    sim = Simulation()
    load_agents(sim)




def load_agents(sim):

    for i in range(sim.TOTAL_POPULATION):
        sim.activeAgents.append(Agent(initBrain=False, initEmpty=False))



def draw():
    sim = Simulation()
    print(sim.blockCounter)

    for c in range(sim.CYCLES):

        if sim.blockCounter % 75 == 0:
            sim.blocks.append(Block())
        sim.blockCounter = sim.blockCounter + 1

        i = 0
        while i < len(sim.blocks):
            sim.blocks[i].update()

            if sim.blocks[i].offscreen():
                sim.blocks.pop(i)
                i = i - 1

            i = i + 1

        #all agents will have the same xPos
        globalXPos = sim.activeAgents[0].xPos

        # find closestblock. all agents have same x pos and it never changes so the closest block
        # for one will be the closest block for all

        closestblock = None

        for i in range(len(sim.blocks)):
            if globalXPos < sim.blocks[i].xPos + sim.blocks[i].width:
                closestblock = sim.blocks[i]
                break

        # Go through each agent check if they have hot something and if they have add them to deadAgents list
        i = 0
        while i < len(sim.activeAgents):

            agent = sim.activeAgents[i]
            agent.think(closestblock)
            agent.update(closestblock)

            if closestblock.hit(agent):
                # Now we penelise the agent based on their average distance from the center of the gaps
                # over the course of its entire life. this is to help distinguish between performance
                # of agents that all crash into the same block, the ones that were closer to the gap,
                # were on a better track than the ones away from it

                agent.computeFitness()
                sim.deadAgents.append(agent)
                sim.activeAgents.pop(i)
                i = i - 1

            i = i + 1


        # if everything is dead restart the game

        if len(sim.activeAgents) == 0:
            sim.activeAgents = GA.produceNextGeneration(sim.deadAgents)
            sim.generationCount = sim.generationCount + 1


        # draw everything
        background(255)

        for i in range(len(sim.blocks)):
            sim.blocks[i].show()

        for i in range(len(sim.activeAgents)):
            sim.activeAgents[i].show()


run()
