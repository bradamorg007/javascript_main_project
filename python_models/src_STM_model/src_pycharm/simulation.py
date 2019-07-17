import pygame
from level_manager import LevelManager
from agent_manager import AgentManager
from neuroevolution import NeuroEvolution
from image_capture import ImageCapture


def run(population_size, cycles, level_manager, image_capture):
    # Call this function so the Pygame library can initialize itself
    pygame.init()

    # Create an 800x600 sized screen
    screen = pygame.display.set_mode([level_manager.SCREEN_WIDTH, level_manager.SCREEN_HEIGHT])

    # Set the title of the window
    pygame.display.set_caption('Simulation Session')

    # Define Agents array sprite array dead agents array
    #active_agents sprites for drawing, active agents none sprites for computation
    agents = AgentManager(population_size=population_size, level_manager=level_manager)

    neuro_evolution = NeuroEvolution()

    # Define Block array normal array with sprites in it
    active_blocks = []


    # Main simulation loop
    clock = pygame.time.Clock()
    frame_count = 0
    END_SIM = False
    time_start = pygame.time.get_ticks()
    while not END_SIM:

        # --- Event Processing ---

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                END_SIM = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    END_SIM = True


        # cyles loop will capture all the game logic
        label = ''
        for i in range(cycles):

            # --- Game Logic ---
            # house keeping on block objects
            active_blocks = level_manager.monitor(active_blocks, frame_count)

            # update block positions
            # create current labels present in active_blocks
            for i in range(len(active_blocks)):
                active_blocks[i].update()
                label += active_blocks[i].class_label + '-'


            level_manager, agents, active_blocks = neuro_evolution.run(level_manager, agents, active_blocks)

        if level_manager.TERMINATE_FLAG:
            print('\n simulation is ending')
            print()
            if level_manager.TERMINATE_FLAG and not level_manager.END_SIMULATION_FLAG:

                print('Termination Cause: The frequency of block object occurrences \n '
                      'is too small to occupy the screen at all times given the simulation time length. \n '
                      'This will cause xPos not found error during training. Recommend decreasing simulation \n'
                      ' time length or increase number of block objects ')


            elif level_manager.TERMINATE_FLAG and  level_manager.END_SIMULATION_FLAG:
                print('Termination Cause: Normal end of epoch termination')

            print()
            END_SIM = True

        # --- Drawing ---
        screen.fill(level_manager.colors.WHITE)

        for block in active_blocks:
            block.draw(screen)

        agents.draw(screen, mode=level_manager.mode)

        image_capture.capture(surface=screen, label=label, level_manager=level_manager, frame_count=frame_count)

        pygame.display.flip()
        agents.clear(mode=level_manager.mode)


        if level_manager.RESET_FLAG:

            frame_count = 0
            level_manager.RESET_FLAG = False
        else:
            frame_count += 1


        clock.tick(level_manager.FPS)

    time_end = pygame.time.get_ticks()
    print('Simulation Terminated')
    print('Simulation Time Length = %s' % ((time_end-time_start)/ 1000))
    pygame.quit()


if __name__ == "__main__":

    POPULATION_SIZE = 200
    CYLES = 1 # 1-200
    DRAW_SIM = True

    SCREEN_HEIGHT = 600
    SCREEN_WIDTH = 800

    blueprints = [[50, 150, None, 100, 4, 49, 'unseen'],
                  [400, 500, None, 100, 4, 51, 'seen'],
                  ]

    LEVEL_MANAGER = LevelManager(FPS=60,
                                 game_len=300,
                                 epochs=2,
                                 number_of_blocks=100,
                                 buffer_size=10,
                                 blueprints=blueprints,

                                 override_gap_size=100,
                                 override_block_width=100,
                                 override_block_speed=4,

                                 data_augmentation=True,
                                 y_top_jitter_probability=0.5,
                                 y_top_jitter_amount=50,
                                 y_bottom_jitter_probability=0.5,
                                 y_bottom_jitter_amount=50,
                                 width_jitter_probability=0.5,
                                 width_jitter_amount=40,

                                 batch_reset='seed',
                                 screen_dimensions=(SCREEN_WIDTH, SCREEN_HEIGHT),
                                 optional_script_build_args='percentage',
                                 mode='capture', capture_mode_override=False
                                 )

    # init image capture
    max_frames = LEVEL_MANAGER.compute_max_frames(capture_first_epoch_only=True)
    filename = 'data_seen_unseen_dynamic'

    IMAGE_CAPTURE = ImageCapture(buffer_size=0.05, max_frames=max_frames, step_size=1,
                                 capture_first_epoch_only=True, capture_mode='save',
                                 save_folder_path=filename, grey_scale=True,
                                 rescale_shape=(40, 40), normalise=False, preview_images=False, show_progress=True)

    LEVEL_MANAGER.save_config(save_folder_path=filename)
    IMAGE_CAPTURE.save_config(save_folder_path=filename)

    run(POPULATION_SIZE, CYLES, LEVEL_MANAGER, IMAGE_CAPTURE)