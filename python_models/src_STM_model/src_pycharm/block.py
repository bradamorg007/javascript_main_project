
from p5 import *
import numpy as np

class Block:

    def __init__(self):

        self.width = 100
        self.speed = 4
        self.xPos = width
        gapSize = 80

        topStart = np.random.rand() * (height - gapSize)
        min = topStart + gapSize
        bottomStart = np.floor(np.random.rand() * (height - min + 1)) + min

        self.topStart = topStart
        self.bottomStart = bottomStart


    def show(self):

        fill(0)
        rect(self.xPos, 0, self.width, self.topStart)
        rect(self.xPos, self.bottomStart, self.width, height)


    def update(self):
        self.xPos = self.xPos - self.speed


    def hit(self, agent):

        if agent.yPos - (agent.radius * 0.5) < self.topStart or agent.yPos + (agent.radius * 0.5) > self.bottomStart:
            if (agent.xPos > self.xPos) and agent.xPos < self.xPos + self.width:

                agent.yPos = height/2
                return True

        return False


    def offscreen(self):
        if self.xPos < -self.width:
            return True
        else:
            return False


if __name__ == "__main__":
    pass




