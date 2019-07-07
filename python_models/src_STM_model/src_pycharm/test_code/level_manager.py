from test_code.block import Block
import numpy as np
import pygame

class Colors:

    def __init__(self):
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED =   (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE =  (0, 0, 255)


class LevelManager():

    def __init__(self, FPS=60, game_len=300, epochs=10, number_of_blocks=100,
                 buffer_size=10, blueprints=None, overide_gap_size=None,
                 block_width=100, block_speed=4,
                 screen_dimensions=(300, 300), batch_reset='seed',
                 optional_script_build_args="percentage"):

        self.colors = Colors()
        self.FPS = FPS
        self.game_len = game_len
        self.epochs = epochs
        self.epoch_count = 0
        self.block_width = block_width
        self.block_speed = block_speed

        self.buffer_size = buffer_size
        self.buffer = []
        self.batch_reset = batch_reset
        self.level_script = []
        self.level_script_bin = []

        self.number_of_blocks = number_of_blocks
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = screen_dimensions
        self.block_display_freq_per_epoch = np.floor((self.FPS * self.game_len) / self.number_of_blocks)
        self.END_SIMULATION_FLAG = False

        pattern = ['ordered', "percentage",'percentage_ordered' ]

        match = False
        for p in pattern:
            if p == optional_script_build_args:
                match = True
                break

        if match == False:
            self.optional_script_build_args = "percentage"
        else:
            self.optional_script_build_args = optional_script_build_args


        sum = None
        if blueprints == None:
            # use predefined blueprint
            # blueprint made of y_top, y_bottom, max_gapSizem, width, speed, percentage occrance

            blueprints = [[150, 200,  None, 100, 4, 50],
                         [50,   150,  None, 100, 4, 10],
                         ['r',  'r',  80,   50,  4, 10],
                         [200,  270,  None, 100, 4, 10],
                         [250,  300,  None, 100, 4, 10],
                         [0,    80,   None, 100, 4, 10]]

            sum = self.blueprint_validation(blueprints)
            self.blueprints = blueprints


        elif blueprints == 'random':
            self.blueprints = blueprints

            if overide_gap_size == None:
                self.overide_gape_size = np.random.randint(0, self.SCREEN_HEIGHT, size=1)

            else:
                self.overide_gape_size = overide_gap_size

        else:

            sum = self.blueprint_validation(blueprints)
            self.blueprints = blueprints

        self.sum = sum

        # build level script and load in first batch to buffer
        self.build_level_script(sum)
        self.buffer_to_script_request()


    def build_level_script(self, sum):
        # build the level script based on the blueprints

        if self.blueprints == 'random':
            r = ['random'] * 6
            for i in range(self.number_of_blocks):
                self.level_script.append(r)
        else:

            if self.optional_script_build_args == "ordered":
                for i in range(self.number_of_blocks):
                    for blueprint in self.blueprints:
                        self.level_script.append(blueprint)

            elif self.optional_script_build_args == "percentage":
                for i in range(self.number_of_blocks):
                        self.level_script.append(self.selector(self.blueprints, sum))

            elif self.optional_script_build_args == 'percentage_ordered':
                for i in range(self.number_of_blocks):
                        self.level_script.append(self.selector(self.blueprints, sum))

                # sort based on percentage in descending order elemets with highest percentage first
                func = lambda x : x[5]
                self.level_script.sort(key=func, reverse=True)


    def level_script_pull_request(self):
        blueprint = self.level_script.pop(0)

        if self.batch_reset == 'seed':
            self.level_script_bin.append(blueprint)

        return blueprint


    def buffer_to_script_request(self):
        # if buffer is empty it will fill it.
        # elif it will fill all available positions up to the buffer size
        # else if buffer is full it will pop first element off and new item to end like a que

        def append_buffer(loop_size):

            for i in range(loop_size):

                blueprint = self.level_script_pull_request()
                block_obj = blueprint_reader(blueprint)

                if block_obj == None:
                    raise ValueError(
                        'ERROR: buffer to script request: blueprint %s contains unreadable input signitures' % i)

                self.buffer.append(block_obj)


        def blueprint_reader(blueprint):

            y_top, y_bottom, max_gapSize, width, speed, _ = blueprint

            # make the block generically
            if width == 'random' and speed == 'random':
                width = self.block_width
                speed = self.block_speed

            block = Block(SCREEN_WIDTH=self.SCREEN_WIDTH, SCREEN_HEIGHT=self.SCREEN_HEIGHT,
                          width=width, speed=speed)

            # select specific build config

            if y_top == 'r':
                block.rand_config(max_gap_size=max_gapSize)
                block.build()

            elif y_top == 'random':
                block.rand_config(max_gap_size=self.overide_gape_size)
                # save random blueprint to level_script bin to seed random generation
                # blueprint made of y_top, y_bottom, max_gapSizem, width, speed, percentage occrance
                if self.batch_reset == 'seed':
                    self.level_script_bin[len(self.level_script_bin) - 1] = [block.topStart,
                                                                             block.bottomStart,
                                                                             self.overide_gape_size,
                                                                             block.width,
                                                                             block.speed, 1
                                                                             ]
                block.build()

            elif isinstance(y_top, int) and isinstance(y_bottom, int):
                block.manual_config(topStart=y_top, bottomStart=y_bottom)
                block.build()

            else:
                return None

            return block


        if len(self.buffer) == 0:

            append_buffer(loop_size=self.buffer_size)

        elif len(self.buffer) < self.buffer_size:

            diff = self.buffer_size - len(self.buffer)
            append_buffer(loop_size=diff)

        elif len(self.buffer) == self.buffer_size:
            pass
           # append_buffer(loop_size=1)

        elif len(self.buffer) > self.buffer_size:

            raise ValueError("ERROR FATAL buffer_to_script_request: current buffer exceeds buffer size limit")

        else:
            raise ValueError("ERROR FATAL buffer_to_script_request: unexpected error detected please investigate")


    def buffer_pull_request(self):

        output_obj = self.buffer.pop(0)
        self.buffer_to_script_request()

        return output_obj


    def selector(self, input, sum):

        index = 0
        r = np.round(np.random.rand() * sum)

        while r > 0:
            r = r - input[index][len(input[index])-1]

            if r > 0:
                index = index + 1

        selection = input[index]

        if selection == None:
            raise ValueError("ERROR GA: Parent in select parent method is undefined this is due to the indexing")

        return selection


    def blueprint_validation(self, blueprints):

        sum = 0
        for blueprint in blueprints:

            if len(blueprint) != 6 and len(blueprint) != 5:
                raise ValueError()

            if blueprint[0] != 'r':

                if blueprint[0] < 0 or blueprint[0] > blueprint[1]:
                    raise ValueError()

                if blueprint[1] > self.SCREEN_HEIGHT or blueprint[1] < blueprint[0]:
                    raise ValueError

                if blueprint[2] != None:
                    raise ValueError('Error BluePrint sytax: predefined object positions do not require manually specified gap size')

            else:

                if blueprint[2] == None:
                    raise ValueError('Error BluePrint sytax position [2]: randomly defined object positions require a manually specified gap size')


            if blueprint[3] < 1:
                raise ValueError()

            if blueprint[5] < 0 or blueprint[5] > 100:
                raise ValueError()

            sum += blueprint[5]

        if sum != 100:
            raise ValueError()

        return sum


    def monitor(self, active_blocks, frame_count):

        # checks when to add new object to game from buffer
        # checks if obj is off screen and needs removing
        # chekcs if reached end of epoch, resest level_script and increments epoch counter

        if len(self.level_script) == 0:

            if self.batch_reset == 'seed':
                self.level_script = self.level_script_bin
                self.level_script_bin = []

            else:
                self.build_level_script(self.sum)
                self.buffer_to_script_request()

            self.epoch_count += 1

        if self.epoch_count > self.epochs - 1:
            self.END_SIMULATION_FLAG = True


        #add new blocks
        if frame_count % self.block_display_freq_per_epoch == 0:
            active_blocks.append(self.buffer_pull_request())

        # remove offscreen blocks
        i = 0
        while i < len(active_blocks):
            if active_blocks[i].offscreen():
                active_blocks.pop(i)
                i -= 1
            i += 1

        frame_count += 1
        return active_blocks, frame_count

    def level_reset(self, active_blocks):

        self.epoch_count = 0

        if self.batch_reset == 'seed':
            # reset level from the start
            if len(self.level_script_bin) > 0:

                # clear the buffer for new batch. the add bin back to level_script
                self.buffer = []
                for i in range(len(self.level_script_bin)-1, -1, -1):
                    self.level_script.insert(0, self.level_script_bin[i])

                self.level_script_bin = []
                # refill buffer
                self.buffer_to_script_request()

            else:
                raise ValueError("ERROR LevelManager level reset: batch reset is True but there is nothing in the levelscript bin to reset with")

        else:
            # if not seed then init a new random init

            self.build_level_script(self.sum)

        return []





if __name__ == "__main__":

    level_manager = LevelManager(FPS=60, game_len=20, epochs=10, number_of_blocks=50,
                                 buffer_size=10, overide_gap_size=80,
                                 batch_reset='seed', optional_script_build_args='percentage_ordered')

    block_obj = level_manager.buffer_pull_request()

    a = 0