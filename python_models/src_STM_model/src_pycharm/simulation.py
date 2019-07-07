import pygame
import numpy as np
from test_code.block import Block
from test_code.level_manager import LevelManager
from nn import NeuralNetwork
from GA import GA
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


    def draw(self, surface):

        self.sprites.add(self.not_sprites)
        self.sprites.draw(surface)


    def clear(self):
        self.sprites = pygame.sprite.Group()


    def update_arrays(self, input):

        if isinstance(input, list):

            self.not_sprites = input
            self.sprites = pygame.sprite.Group()
            self.len = len(self.not_sprites)
            self.dead_agents = []

        else:
            raise ValueError("ERROR ActiveAgents: input must be of type list")


def get_closest_block(active_blocks, global_xPos):
    closest_block = None

    for i in range(len(active_blocks)):
        if global_xPos < active_blocks[i].bottom_block.rect.right:
            closest_block = active_blocks[i]
            active_blocks[i].top_block.image.fill((255, 0, 0))
            break

    return  closest_block



def run(population_size, cycles, draw=True, LM=None):
    # Call this function so the Pygame library can initialize itself
    pygame.init()

    # Create an 800x600 sized screen
    screen = pygame.display.set_mode([LM.SCREEN_WIDTH, LM.SCREEN_HEIGHT])

    # Set the title of the window
    pygame.display.set_caption('Simulation Session')

    # Define Agents array sprite array dead agents array
    #active_agents sprites for drawing, active agents none sprites for computation
    agents = AgentManager(population_size=POPULATION_SIZE, level_manager=LM)

    # Define Block array normal array with sprites in it
    active_blocks = []

    # Main simulation loop
    trig = True
    frame_count = 0
    generation_count = 0
    block_count = 0
    clock = pygame.time.Clock()
    END_SIM = False
    while not END_SIM:

        # --- Event Processing ---

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                END_SIM = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    END_SIM = True

            if LM.END_SIMULATION_FLAG:
                END_SIM = True
                print('simulation has ended')

        # cyles loop will capture all the game logic
        for i in range(cycles):

            # --- Game Logic ---
            # house keeping on block objects
            active_blocks, block_count = LM.monitor(active_blocks, frame_count=frame_count)

            # update block positions
            for i in range(len(active_blocks)):
                active_blocks[i].update()


            # determine closest block
            global_xPos = agents.not_sprites[0].rect.x

            closest_block = get_closest_block(active_blocks, global_xPos)

            # if closest_block == None:
            #     active_blocks.append(LM.buffer_pull_request())
            #     closest_block = get_closest_block(active_blocks, global_xPos)

            # agents think using nn then update their positions
            # check if agents a block. if so remove from active_agents and add to dead agents
            i = 0
            while i < len(agents.not_sprites):

                agent = agents.not_sprites[i]
                agent.think(closest_block, LM.SCREEN_WIDTH, LM.SCREEN_HEIGHT)
                agent.update(closest_block, LM.SCREEN_HEIGHT)

                if closest_block.hit(agent):

                    agent.computeFitness()
                    agents.splice(i)
                    i -= 1

                i += 1

            # check if all active agents are dead, the perform GA and reset game level and epochs
            if len(agents.not_sprites) == 0:
                new_population = GA.produceNextGeneration(population=agents.dead_agents,
                                                          screen_width=LM.SCREEN_WIDTH,
                                                          screen_height=LM.SCREEN_HEIGHT)



                agents.update_arrays(new_population)
                generation_count += 1
                active_blocks = LM.level_reset(active_blocks)
                trig = False
                print('generation = %s population size = %s epoch = %s / %s' % (generation_count, len(new_population), LM.epoch_count, LM.epochs))
                if generation_count == 15:
                    draw = True

                # reset level

                # reset test

            # if frame_count > 500 and trig == True:
            #
            #     LM.level_reset()
            #     trig = False




        # --- Drawing ---

        if draw == True:
            screen.fill(LM.colors.WHITE)

            for block in active_blocks:
                block.draw(screen)

            agents.draw(screen)

            pygame.display.flip()
            agents.clear()

        clock.tick(LM.FPS)
        if trig == True:
            frame_count += 1
        else:
            frame_count = 0
            trig = True

    pygame.quit()



POPULATION_SIZE = 200
CYLES = 1 # 1-200
DRAW_SIM = True

SCREEN_HEIGHT = 60
SCREEN_WIDTH = 60

blueprints = [[5, 20, None, 10, 1, 51],
              [40, 55, None, 10, 1, 49],
              ]

LEVEL_MANAGER = LevelManager(FPS=30,game_len=30, epochs=5, number_of_blocks=20,
                             buffer_size=10, overide_gap_size=20, blueprints=blueprints,
                             block_speed=1, block_width=10,
                             screen_dimensions=(SCREEN_WIDTH, SCREEN_HEIGHT),
                             batch_reset='seed', optional_script_build_args='percentage_ordered')

run(POPULATION_SIZE, CYLES, DRAW_SIM, LEVEL_MANAGER)