import pygame
from agent import Agent

class AgentManager:
    def __init__(self, population_size, level_manager):
        not_sprites = []

        for i in range(population_size):
            not_sprites.append(Agent(initBrain=False, initEmpty=False,
                                           screen_width=level_manager.SCREEN_WIDTH,
                                           screen_height=level_manager.SCREEN_HEIGHT, name=str(i)))


            self.len = len(not_sprites)
            self.sprites = pygame.sprite.Group()
            self.not_sprites = not_sprites
            self.dead_agents = []


    def splice(self, index):

        dead_agent = self.not_sprites.pop(index)
        self.dead_agents.append(dead_agent)
        self.len = len(self.not_sprites)


    def draw(self, surface, mode):

        if mode == 'train':
            self.sprites.add(self.not_sprites)
            self.sprites.draw(surface)


    def clear(self, mode):
        if mode == 'train':
          self.sprites = pygame.sprite.Group()


    def update_arrays(self, input):

        if isinstance(input, list):

            self.not_sprites = input
            self.sprites = pygame.sprite.Group()
            self.len = len(self.not_sprites)
            self.dead_agents = []

        else:
            raise ValueError("ERROR ActiveAgents: input must be of type list")


if __name__ == '__main__':
    pass