from GA import GA

class NeuroEvolution:

    def __init__(self):

        self.generation_count = 0


    def run(self, level_manager, agents, active_blocks):


        if level_manager.mode == 'train':
            # determine closest block
            global_xPos = agents.not_sprites[0].rect.x
            closest_block = self.get_closest_block(active_blocks, global_xPos)

            # agents think using nn then update their positions
            # check if agents a block. if so remove from active_agents and add to dead agents
            i = 0
            while i < len(agents.not_sprites):

                agent = agents.not_sprites[i]
                agent.think(closest_block, level_manager.SCREEN_WIDTH, level_manager.SCREEN_HEIGHT)
                agent.update(closest_block, level_manager.SCREEN_HEIGHT)

                if closest_block.hit(agent):
                    agent.computeFitness()
                    agents.splice(i)
                    i -= 1

                i += 1

            # check if all active agents are dead, the perform GA and reset game level and epochs
            if len(agents.not_sprites) == 0:
                new_population = GA.produceNextGeneration(population=agents.dead_agents,
                                                          screen_width=level_manager.SCREEN_WIDTH,
                                                          screen_height=level_manager.SCREEN_HEIGHT)

                agents.update_arrays(new_population)
                self.generation_count += 1
                active_blocks = level_manager.level_reset(active_blocks)
                level_manager.RESET_FLAG = True
                print('generation = %s population size = %s epoch = %s / %s' % (
                self.generation_count, len(new_population), level_manager.epoch_count, level_manager.epochs))


        return level_manager, agents, active_blocks


    def get_closest_block(self, active_blocks, global_xPos):
        closest_block = None

        for i in range(len(active_blocks)):
            if global_xPos < active_blocks[i].bottom_block.rect.right:
                closest_block = active_blocks[i]
                active_blocks[i].top_block.image.fill((255, 0, 0))
                break

        return closest_block