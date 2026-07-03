import pygame, sys
from pygame.locals import *
import math
from time import sleep
from copy import deepcopy
import numpy as np
import random

from PIL import Image

from .levels.skeleton import *

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

GOLD = (255, 200, 0)

class discreteGame:
    def __init__(self, settings = None, envMode = True, default_colors = True):
        # params for random initialization; usually ignored (put them into a Settings object?)
        self.typically_restrict_angles = False
#        self.typical_indicator_length = 0.5
        self.typical_wall_width = 50/800
        self.side_wall_width = 50/800
        self.typical_min_wall_height = 300/800
        self.typical_max_wall_height = 600 / 800
        self.typical_max_wall_num = 2 # not counting side walls
        self.typical_agent_r = 0.05
        self.typical_gold_r = 1.0/64
        self.typical_max_gold_num = 2
        if settings is None:
            settings = self.random_settings( restrict_angles = self.typically_restrict_angles )

        # End of randomization params
        self.reward = 0
        self.envMode = envMode
        self.initial = deepcopy(settings)

        if default_colors:
            self.BLACK = BLACK
            self.WHITE = WHITE
            self.RED = RED
            self.GREEN = GREEN
            self.BLUE = BLUE
            self.GOLD = GOLD
        else:
            raise NotImplementedError

        self.sprite_files = { 'arrow': "game_images_and_modifications/Arrow_example_for_import.png", \
                              'line' : "game_images_and_modifications/LineSprite.png" }

        self.actions = [(lambda : 0), self.stepForward, self.stepBackward, self.swivel_clock, self.swivel_anticlock]
        self.settings = settings

        if envMode:
            self.windowSurface = pygame.Surface((self.settings.gameSize, self.settings.gameSize))
        else:
            # set up pygame
            pygame.init()
            # set up the window
            self.windowSurface = pygame.display.set_mode((self.settings.gameSize, self.settings.gameSize), 0, 32)
            pygame.display.set_caption('discrete engine')
               
        self.windowSurface.fill(self.WHITE)
        self.universal_update()

        if not self.envMode:
            self.humanGame()

    def reset(self):
        self.settings = deepcopy(self.initial)
        self.reward = 0
        self.universal_update()
        if not self.envMode:
            self.humanGame()
            return None, {}
        else:
            return self.getData(), {} # dummy 'info' dictionary for now.

    def random_reset(self, restrict_angles = False):
        self.initial = self.random_settings(self.settings.gameSize, restrict_angles)
        return self.reset()

    ####### Functions for drawing / evaluating position.
    def backRot(self, pos_x, pos_y, theta): # Counterclockwise, compensating
        c = math.cos(theta)
        s = math.sin(theta)
        return c*pos_x + s*pos_y, 0 - s*pos_x + c*pos_y

    def forwardRot(self, pos_x, pos_y, theta, center_x = 0, center_y = 0):
        """Rotate pos_x, pos_y by theta, forward, around center_x, center_y"""
        rel_x, rel_y = self.backRot(pos_x - center_x, pos_y - center_y, 0 - theta)
        return center_x + rel_x, center_y + rel_y
    
    def top_corner_adjustment(self, orig_x, orig_y, w, h, theta):
        """Used for the correct top corner of the frame containing the wall, 
    so that the (pre-rotation) top-left corner of the WALL is at  orig_x, orig_y.
    That is, the wall can be thought of as drawn with corner at orig_x, orig_y,
    then rotated clockwise through angle theta around this anchoring corner."""
        quadrant = (math.floor(theta / (0.5*math.pi)) % 4 ) + 1 # Clockwise from 
        if quadrant == 1:
            return orig_x - h*math.sin(theta), orig_y
        elif quadrant == 2:
            d = math.sqrt(h**2 + w**2)
            direction_angle = theta - math.atan(w / h)
            return orig_x - d*math.sin(direction_angle), orig_y + h*math.cos(theta) # cosine is negative.
        elif quadrant == 3:
            d = math.sqrt(h**2 + w**2)
            direction_angle = theta - math.atan(w / h)
            return orig_x - w*math.sin(theta - math.pi/2), orig_y + d*math.cos(direction_angle)
        else: # quadrant == 4
            return orig_x, orig_y + w*math.cos(theta - math.pi/2)
    
    def true_coords(self, coords):
        return coords[0]*self.settings.gameSize, coords[1]*self.settings.gameSize

    def direction_angle(self, xo, yo, xt, yt):
        # Finds the angle from (xo, yo) to (xt, yt)
        # Degenerate cases first, just in case
        if xo == xt:
            if yo < yt:
                return math.pi / 2
            else:
                return 3 * math.pi / 2
        if yo == yt:
            if xo < xt:
                return 0.0
            else:
                return math.pi
        candidate = math.atan((yt - yo) / (xt - xo))
        if (yt > yo):
            # Quadrant 1
            if (xt > xo):
                return candidate
            # Quadrant 2
            else:
                return math.pi + candidate # candidate is negative in this case
        else:
            # Quadrant 3
            if (xt < xo):
                return math.pi + candidate # candidate is positive in this case.
            # Quadrant 4
            else:
                return (2 * math.pi) + candidate # candidate is negative

    def draw_arrow(self, extension = 1.0, direction = None, sprite = None):
        """Drawing arrows is an important cognitive tool / middle step that I will teach the agent.
        The agent will also have the option to extend the arrow, for example to reach an object.
        Note that this will be taught to the agent as a skill based on verbal prompts; this image generation
        is purely for the sake of drawing 'target' images, which the agent will be trained on."""
        if direction is None:
            direction = self.settings.direction
        if sprite is None:
            sprite = 'line'
        # in the future, I may choose a different asset.
        FPATH = self.sprite_files[sprite]
        
        arrowImage = pygame.image.load(FPATH)
        im_length, im_width = arrowImage.get_size()
        default_length = self.settings.agent_r * self.settings.gameSize * 2
        ratio = default_length / im_length
        arrowImage = pygame.transform.scale(arrowImage, (ratio * im_length * extension, ratio * im_width))
        offset = arrowImage.get_size()[1] / 2 # offset from the middle of the top of the arrow
        mid_x = (self.settings.agent_x + (1.0 * self.settings.agent_r * math.cos(direction))) * self.settings.gameSize
        mid_y = (self.settings.agent_y + (1.0 * self.settings.agent_r * math.sin(direction))) * self.settings.gameSize
        # This is the position around which the arrow needs to be rotated to end up in the correct place
        orig_x = mid_x + offset*math.sin(direction)
        orig_y = mid_y - offset*math.cos(direction)
        length, width = arrowImage.get_size()
        # Same adjustment as the walls
        tc_x, tc_y = self.top_corner_adjustment(orig_x, orig_y, length, width, direction)
        arrowImage = pygame.transform.rotate(arrowImage, 0 - direction * 180 / math.pi)
        self.windowSurface.blit(arrowImage, (tc_x, tc_y))

    def draw_corners(self, tp, offset = None, radius = None):
        # tp is true wall params
        if offset is None:
            offset = self.settings.agent_r * 1.2 * self.settings.gameSize
        if radius is None:
            radius = self.settings.gold_r * self.settings.gameSize # why not? change me if needed
        factor = math.sqrt(0.5) # precomputed during compilation, I think, so faster code
        offset_x = offset * factor # math.cos(math.pi / 4)
        offset_y = offset * factor # math.sin(math.pi / 4)
        # centers before wall rotation.
        og_centers = []
        # top left
        og_centers.append((tp[0] - offset_x, tp[1] - offset_y))
        # top right
        og_centers.append((tp[0] - offset_x, tp[1] + tp[3] + offset_y))
        # bottom left
        og_centers.append((tp[0] + tp[2] + offset_x, tp[1] - offset_y))
        # bottom right
        og_centers.append((tp[0] + tp[2] + offset_x, tp[1] + tp[3] + offset_y))
        centers = [self.forwardRot(center[0], center[1], tp[4], tp[0], tp[1]) for center in og_centers]
        for center in centers:
            pygame.draw.circle(self.windowSurface, \
                               self.BLUE, \
                               center, \
                               radius)

    def draw_agent(self):
        # I *could* move this to the init function, but I won't for now.
        # If CPU computation becomes a problem, that's an easy optimization
        agent_x = self.settings.agent_x * self.settings.gameSize
        agent_y = self.settings.agent_y * self.settings.gameSize
        agent_r = self.settings.agent_r * self.settings.gameSize
#        indicator_length = self.settings.indicator_length * self.settings.gameSize
        eye_x = (self.settings.agent_x + (0.6 * self.settings.agent_r * math.cos(self.settings.direction))) * self.settings.gameSize
        eye_y = (self.settings.agent_y + (0.6 * self.settings.agent_r * math.sin(self.settings.direction))) * self.settings.gameSize
        eye_r = 0.4 * agent_r
        pygame.draw.circle(self.windowSurface, \
                           self.GREEN, \
                           (agent_x, agent_y), \
                           agent_r)
        pygame.draw.circle(self.windowSurface, \
                           self.RED, \
                           (eye_x, eye_y), \
                           eye_r)
#        pygame.draw.line(self.windowSurface, \
#                         self.BLACK, 
#                         (agent_x, agent_y), (agent_x + math.cos(self.settings.direction)*indicator_length, agent_y + math.sin(self.settings.direction)*indicator_length))
    
    def draw_gold(self):
        gold_r = self.settings.gold_r * self.settings.gameSize
        for coords in self.settings.gold:
            tc = self.true_coords(coords)
            pygame.draw.circle(self.windowSurface, self.GOLD, tc, gold_r)

    def true_wall_params(self, params):
        tp = [val * self.settings.gameSize for val in params[:4]]
        tp.append(params[4]) # angle treated differently
        return tp
    
    def draw_walls(self):
        for params in self.settings.walls:
            tp = self.true_wall_params(params)
            clientSurface = pygame.Surface((tp[2], tp[3]))
            clientSurface.fill(self.WHITE)
            clientSurface.set_colorkey(self.WHITE)
#            clientSurface = clientSurface.convert_alpha(self.windowSurface)
            pygame.draw.rect(clientSurface, self.BLACK, (0, 0, tp[2], tp[3]))
            clientSurface = pygame.transform.rotate(clientSurface, 0 - params[4]*180/math.pi) # Format is consistent with js
            newX, newY = self.top_corner_adjustment(tp[0], tp[1], tp[2], tp[3], tp[4])
            self.windowSurface.blit(clientSurface, (newX, newY)) 
    
    def draw(self, ignore_agent=False, ignore_gold=False, ignore_walls=False):
        self.windowSurface.fill(self.WHITE)
        if not ignore_agent:
            self.draw_agent()
        if not ignore_walls:
            self.draw_walls()
        if not ignore_gold:
            self.draw_gold()
        if not self.envMode:
            pygame.display.update()
    
    ####### Overlap detection / updating function
    def mod2pi(self, theta):
        rotationAngle = math.floor(theta/(2*math.pi))*2*math.pi
        return theta-rotationAngle
    
    def spot_overlap_check(self, x, y, spot_x, spot_y, spot_r, agent_r=None):
        if agent_r is None:
            agent_r = self.settings.agent_r # typical case; also used for generating random games, hence the ambiguity.
        pointing = (x - spot_x, y - spot_y)
        overlap = math.sqrt(pointing[0]**2 + pointing[1]**2) - agent_r - spot_r
        if (overlap < 0):
            return True
        else:
            return False
    
    def gold_update(self):
        # Going backward to prevent the deletions from affecting the traversal
        collected = 0
        for i in range(len(self.settings.gold) -1, -1, -1):
            if (self.spot_overlap_check(self.settings.agent_x, \
                                   self.settings.agent_y, \
                                   self.settings.gold[i][0], \
                                   self.settings.gold[i][1], \
                                   self.settings.gold_r)):
                del self.settings.gold[i]
                self.reward += 1;
                collected += 1
                #print("Reward: " + str(self.reward));
        return collected
    
    def universal_update(self):
        collected = self.gold_update()
        self.draw()
        if not self.envMode:
            sleep(1.0/10)
        return collected
    
    def wall_overlap_check(self, old_agent_x, old_agent_y, wall_x, wall_y, wall_w, wall_h, wall_theta, agent_r = None):
        if agent_r is None:
            agent_r = self.settings.agent_r # we can use this to test gold placement for randomly generated levels, hence why different r's
        agent_x, agent_y = self.backRot(old_agent_x, old_agent_y, wall_theta)
        left_lim, top_lim = self.backRot(wall_x, wall_y, wall_theta)
        right_lim = left_lim + wall_w
        bot_lim = top_lim + wall_h
        if ((agent_y >= top_lim) and (agent_y <= bot_lim) and (agent_x >= left_lim) and (agent_x <= right_lim)): # Exotic case, agent inside wall
            return True
        elif ((agent_y >= top_lim) and (agent_y <= bot_lim) and (agent_x <= left_lim) and (agent_x + agent_r > left_lim)): # Hitting from the left
            return True
        elif ((agent_y >= top_lim) and (agent_y <= bot_lim) and (agent_x >= right_lim) and (agent_x - agent_r < right_lim)): # Hitting from the right
            return True
        elif ((agent_x >= left_lim) and (agent_x <= right_lim) and (agent_y <= top_lim) and (agent_y + agent_r > top_lim)): # Hitting from the top
            return True
        elif ((agent_x >= left_lim) and (agent_x <= right_lim) and (agent_y >= bot_lim) and (agent_y - agent_r < bot_lim)): # Hitting from the bottom 
            return True
        elif (self.spot_overlap_check(agent_x, agent_y, left_lim, top_lim, 0, agent_r)): # 4 corner checks 
            return True
        elif (self.spot_overlap_check(agent_x, agent_y, right_lim, top_lim, 0, agent_r)):
            return True
        elif (self.spot_overlap_check(agent_x, agent_y, left_lim, bot_lim, 0, agent_r)):
            return True
        elif (self.spot_overlap_check(agent_x, agent_y, right_lim, bot_lim, 0, agent_r)):
            return True
        else:
            return False
      
    def full_wall_check(self, test_x, test_y, walls=None, agent_r=None):
        if walls is None: # This is also used for random level generation, placing both gold and the agent, hence the ambiguity
            walls = self.settings.walls
        for params in walls:
            if self.wall_overlap_check(test_x, test_y, params[0], params[1], params[2], params[3], params[4], agent_r):
                return False
        return True
    
    # biggest possible step; performed in the original coordinates, NOT in the full, pixel-scale coordinates.
    def biggest_step(self, lim, coords_from_step, min_step=None):
        if min_step is None:
            min_step = 1.0/self.settings.gameSize
        step = lim
        while step > 0:
            test_x, test_y = coords_from_step(step)
            if self.full_wall_check(test_x, test_y):
                return step
            step -= min_step
        return 0
    
    ## Full definition of actions from here.    
    def stepForward(self, lim=None):
        if lim is None:
            lim = 1.0/16 # big enough for most pixelations, small enough to make gameSize 800 interesting.
        stepSize = self.biggest_step(lim, lambda step : (self.settings.agent_x + step*math.cos(self.settings.direction), self.settings.agent_y + step*math.sin(self.settings.direction)))
        self.settings.agent_x += stepSize*math.cos(self.settings.direction)
        self.settings.agent_y += stepSize*math.sin(self.settings.direction)
        return self.universal_update() # returns the gold collected this step.
    
    def stepBackward(self, lim=None):
        if lim is None:
            lim = 1.0/16 # big enough for most pixelations, small enough to make gameSize 800 interesting.
        stepSize = self.biggest_step(lim, lambda step : (self.settings.agent_x - step*math.cos(self.settings.direction), self.settings.agent_y - step*math.sin(self.settings.direction)))
        self.settings.agent_x -= stepSize*math.cos(self.settings.direction)
        self.settings.agent_y -= stepSize*math.sin(self.settings.direction)
        return self.universal_update()
    
    def swivel_anticlock(self):
        self.settings.direction = self.mod2pi(self.settings.direction + math.pi/30)
        return self.universal_update()
    
    def swivel_clock(self):
        self.settings.direction = self.mod2pi(self.settings.direction - math.pi/30)
        return self.universal_update()

    ####### Function for "Arcade" UI   
    def humanGame(self):    
        assert (not self.envMode), "initialize with envMode = False to play"

        # Code to actually update everything
        self.draw()
        pygame.display.update() 

        while True:
            keys=pygame.key.get_pressed()
            if keys[K_LEFT]:
                self.swivel_clock()
            if keys[K_RIGHT]:
                self.swivel_anticlock()
            if keys[K_UP]:
                self.stepForward()
            if keys[K_DOWN]:
                self.stepBackward() 
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return None

    ####### Functions for machine UI: numpy arrays and zoomed-in numpy arrays as output.
    def step(self, actionIndex):
        reward = self.actions[actionIndex]()
        obs = self.getData()
        terminated = False # dummies for now
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def getData(self):
        return pygame.surfarray.array3d(self.windowSurface)/255

    def blowup(self, factor):
        bigSettings = deepcopy(self.settings)
        bigSettings.gameSize = int(factor*self.settings.gameSize)
        slave = discreteGame(bigSettings, envMode=True)
        return slave.getData()

    def _zoom_helper(self, center, factor, canvas):
        bigSize = int(factor*self.settings.gameSize)
        maxCenterCoord = int(bigSize - (self.settings.gameSize/2))
        centerX = min(int(center[0]*factor*self.settings.gameSize), maxCenterCoord)
        centerY = min(int(center[1]*factor*self.settings.gameSize), maxCenterCoord)
        leftPoint = max(int(centerX - (self.settings.gameSize / 2)), 0)
        topPoint = max(int(centerY - (self.settings.gameSize / 2)), 0)
        return canvas[leftPoint:leftPoint + self.settings.gameSize, topPoint:topPoint + self.settings.gameSize]

    def zoom(self, centers, factor):
        assert factor >= 1, "factor must be larger than 1.9"
        canvas = self.blowup(factor) # this part may be slow; a better function would only draw what's in frame.
        batch = np.zeros((len(centers), self.settings.gameSize, self.settings.gameSize, 3))
        for i in range(len(centers)):
            batch[i] = self._zoom_helper(centers[i], factor, canvas)
        return batch

    def random_jitter(self):
        num_actions = 3
        max_num_repeats = 30
        for _ in range(num_actions):
            action_ind = random.randint(1, 4) # skip the 'do nothing' action.
            for _ in range(random.randint(1, max_num_repeats)):
                self.actions[action_ind]()
            
           
    def random_center_near(self, point, scale=None):
        if scale is None:
            scale = self.settings.gold_r
        return (random.uniform(point[0] - scale, point[0] + scale), random.uniform(point[1] - scale, point[1] + scale))

    def corners(self, wall_x, wall_y, wall_w, wall_h, wall_theta):
        c = math.cos( 0 - wall_theta )
        s = math.sin( 0 - wall_theta )
        width_offset_x = wall_w * c
        width_offset_y = wall_w * s
        height_offset_x = wall_h * s # CHECK ME!
        height_offset_y = wall_h * c

        ul = (wall_x, wall_y)
        ur = (wall_x + width_offset_x, wall_y + width_offset_y)
        lr = (wall_x + width_offset_x + height_offset_x, wall_y + width_offset_y + height_offset_y)
        ll = (wall_x + height_offset_x, wall_y + height_offset_y)

        return [ul, ur, lr, ll]

    def random_point_on_line(self, a, b):
        val = random.random()
        nval = 1 - val
        return (a[0] * val + b[0] * nval, a[1] * val + b[1] * nval)

    def random_point_in_quadrilateral(self, corners):
        vals = [random.random() for i in range(4)]
        s = sum(vals)
        fracs = [v / s for v in vals]
        x = sum([fracs[i] * corners[i][0] for i in range(4)])
        y = sum([fracs[i] * corners[i][1] for i in range(4)])
        return (x, y)

    def random_wall_points(self, wall_params, num_points = 4):
        corners = self.corners(*wall_params)
        
        corner_probability = 0.8 # By far the hardest task, and the most important
        wall_probability = 0.15 # No need to go for wall interior too often, nothing there.

        points = []
        for i in range(num_points):
            val = random.random()
            if val < corner_probability:
                points.append(corners[random.randrange(0, 4)])
            elif val < corner_probability + wall_probability:
                ind1 = random.randrange(0, 4)
                ind2 = (ind1 + 1) % 4
                points.append(self.random_point_on_line(corners[ind1], corners[ind2]))
            else:
                points.append(self.random_point_in_quadrilateral(corners))
        return points

    def random_wall_centers(self, num_walls=2, num_each=2):
        points = []
        for i in range(num_walls):
            wall = random.choice(self.settings.walls)
            for point in self.random_wall_points(wall, num_each):
                points.append(point)
        return points

    def random_zoom_center(self, factor):
        offset = 1 / (2*factor)
        maxVal = 1 - offset
        x = random.uniform(offset, maxVal)
        y = random.uniform(offset, maxVal)
        return (x, y)

    # THe below 2 should really be rolled into one
    # full_image_batch is generated for training the NNs; small image batch is for other, specialized purposes
    def random_full_image_batch(self):
        """Batch of ML training images. Full picture, and zooms to some random / important places, at different magnifications"""
        num_factors = 2 # Total number of zoom factors to be used.
        num_gold = 2
        num_agent = 2
        num_walls = 2 # walls sampled for random points
        num_per_wall = 1 # random points per wall
        num_random = 3 # For each scale, 4 more random zooms will be included

        num_per_factor = num_gold + num_agent + num_walls*num_per_wall + num_random
        num_total = 1 + num_factors*num_per_factor + 1 # one normal, lots of closeups, one after jitter
        
        agent_centers = [(self.settings.agent_x, self.settings.agent_y)]
        for i in range(num_agent - 1):
            agent_centers.append(self.random_center_near(agent_centers[0], scale=self.settings.agent_r))

        # In rare cases, we eat the gold as soon as it's created. In that case, golc_centers will just be more points near the agent.
        if len(self.settings.gold) > 0:
            gold_centers = [random.choice(self.settings.gold) for i in range(num_gold)]
        else:
            gold_centers = []
            for i in range(num_gold):
                gold_centers.append(self.random_center_near(agent_centers[0], scale=self.settings.agent_r))

        wall_centers = self.random_wall_centers(num_walls, num_per_wall)

        rand_factor1 = random.uniform(2, 1/(4*self.settings.agent_r))
        rand_factor2 = random.uniform(3, (1/(3*self.settings.gold_r)))

        fac1List = gold_centers + agent_centers + wall_centers
        fac2List = gold_centers + agent_centers + wall_centers

        for i in range(num_random):
            fac1List.append(self.random_zoom_center(rand_factor1))
            fac2List.append(self.random_zoom_center(rand_factor2))

        batch = np.zeros((num_total, self.settings.gameSize, self.settings.gameSize, 3))
        batch[0] = self.getData()
        batch[1:(num_per_factor+1)] = self.zoom(fac1List, rand_factor1)
        batch[(num_per_factor+1):-1] = self.zoom(fac2List, rand_factor2)

        self.random_jitter()
        batch[-1] = self.getData()

        return batch

    def random_small_image_batch(self):
        """just the original and some jitter"""
        num_total = 2
        batch = np.zeros((num_total, self.settings.gameSize, self.settings.gameSize, 3))
        batch[0] = self.getData()
        self.random_jitter()
        batch[-1] = self.getData()
        return batch

    def random_full_image_set(self, numBatches=2, restrict_angles=False):
        """numBatches = full size / 20. Make it divisible by 20.
        Careful using this; this deletes the original game."""
        res = np.zeros((numBatches*20, self.settings.gameSize, self.settings.gameSize, 3))
        ind = 0
        for batch in range(numBatches):
            if batch % 2 == 0:
                for sec in range(10):
                    self.random_reset(restrict_angles)
                    res[ind:ind+2] = self.random_small_image_batch()
                    ind += 2
            else:
                self.random_reset(restrict_angles)
                res[ind:ind+20] = self.random_full_image_batch()
                ind += 20
        return res

    def save_array(self, array, dirname, filename):
        im = Image.fromarray(np.uint8(array * 255), mode='RGB')
        im.save(dirname + '/' + filename + '.png')
        return 0

    def save_screen(self, dirname, filename):
        return self.save_array(self.getData(), dirname, filename)

    def save_image_set(self, image_set, dirname, filebase='game_snapshots'):
        num_files = image_set.shape[0]
        for i in range(num_files):
            self.save_array(image_set[i], dirname, filebase + '_number' + str(i))
        return 0
      
    ####### Functions for random initialization
    def random_ul_corner(self, wall_w, wall_h, wall_theta):
        corners = self.corners(0, 0, wall_w, wall_h, wall_theta)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        mx = min(xs)
        Mx = max(xs)
        my = min(ys)
        My = max(ys)
        toplim = 1.0 - self.side_wall_width - My
        rightlim = 1.0 - self.side_wall_width - Mx
        botlim = self.side_wall_width - my
        leftlim = self.side_wall_width - mx
        return random.uniform(leftlim, rightlim), random.uniform(botlim, toplim)

    def random_wall(self, restrict_angles=False):
        wall_w = self.typical_wall_width
        wall_h = random.uniform(self.typical_min_wall_height, self.typical_max_wall_height)
        if restrict_angles:
            wall_theta = random.randint(0, 1)*math.pi/2 # restrict it to right angles, to make it easier on the autoencoder
        else:
            wall_theta = random.uniform(0, 2*math.pi) # probably overkill, I don't think the symmetries matter for computational efficiency, though.
        wall_x, wall_y = self.random_ul_corner(wall_w, wall_h, wall_theta)
        return [wall_x, wall_y, wall_w, wall_h, wall_theta]

    def random_side_walls(self):
        walls = []
        probability_exit = 0.5
        if random.random() < probability_exit:
            exit_wall = random.randint(0, 3)
        else:
            exit_wall = -1
        for i in range(4):# left wall; top wall; bottom wall; right wall
            if i == exit_wall:
                longside = 0.5 - self.typical_agent_r
            else:
                longside = 1.0
            wall_theta = 0
            isTop = (i != 2) # all but bottom wall have ul on top.
            isLeft = (i < 3) # all but right wall have ul on left
            isHorizontal = ((i == 1) or (i == 2)) # I guess it could be cleaner . . .
            if isLeft:
                wall_x = 0
            else:
                wall_x = 1.0 - self.side_wall_width
            if isTop:
                wall_y = 0
            else:
                wall_y = 1.0 - self.side_wall_width
            if isHorizontal:
                wall_w = longside
                wall_h = self.side_wall_width
            else:
                wall_h = longside
                wall_w = self.side_wall_width
            walls.append([wall_x, wall_y, wall_w, wall_h, wall_theta])
            if i == exit_wall:
                if isHorizontal:
                    wall_y2 = wall_y
                    wall_x2 = longside + 2*self.typical_agent_r
                else:
                    wall_x2 = wall_x
                    wall_y2 = longside + 2*self.typical_agent_r
                walls.append([wall_x2, wall_y2, wall_w, wall_h, wall_theta])
        return walls

    def random_walls(self, restrict_angles=False, num_extra_walls=-1):
        if num_extra_walls < 0:
            num_extra_walls = self.typical_max_wall_num
        walls = self.random_side_walls()
        if num_extra_walls > 0:
            for i in range(random.randint(1, num_extra_walls)):
                walls.append(self.random_wall(restrict_angles))
        return walls

    def random_valid_coords(self, walls, radius, minX = 0.0, maxX = 1.0, minY = 0.0, maxY = 1.0):
        valid = False
        minX = max(minX, self.side_wall_width)
        maxX = min(maxX, 1.0 - self.side_wall_width)
        minY = max(minY, self.side_wall_width)
        maxY = min(maxY, 1.0 - self.side_wall_width)
        while not valid:
            test_x = random.uniform(minX, maxX)
            test_y = random.uniform(minY, maxY)
            valid = self.full_wall_check(test_x, test_y, walls, radius)
        return (test_x, test_y)

    def random_gold(self, walls, max_num_gold = -1, max_agent_offset = 2.0, agent_x = 0.5, agent_y = 0.5):
        if max_num_gold < 0:
            max_num_gold = self.typical_max_gold_num
        gold = []
        if max_num_gold > 0:
            minX = agent_x - max_agent_offset # usually so permissive that the gold can be anywhere
            maxX = agent_x + max_agent_offset
            minY = agent_y - max_agent_offset
            maxY = agent_y + max_agent_offset
            num_gold = random.randint(1, max_num_gold)
            for i in range(num_gold):
                gold.append(self.random_valid_coords(walls, self.typical_gold_r, minX, maxX, minY, maxY))
        return gold

    def random_settings(self, gameSize=64, restrict_angles=False):
        walls = self.random_walls(restrict_angles)
        gold = self.random_gold(walls)
        agent_x, agent_y = self.random_valid_coords(walls, self.typical_agent_r)
        direction = random.uniform(0, 2*math.pi)
        res = Settings(gameSize=gameSize,
#                       indicator_length = self.typical_indicator_length,
                       agent_r = self.typical_agent_r,
                       gold_r = self.typical_gold_r,
                       walls = walls,
                       gold = gold,
                       agent_x = agent_x,
                       agent_y = agent_y,
                       direction = direction)
        return res

    # bare game. Only valid side walls, agent, and 1 gold piece, near the agent
    # basically a tutorial level; will train the agent, initially, on this setup.
    def random_bare_settings(self, gameSize=64, max_agent_offset = -1):
        if max_agent_offset < self.typical_agent_r:
            max_agent_offset = 2.0 * self.typical_agent_r
        walls = self.random_walls(num_extra_walls=0)
        agent_x, agent_y = self.random_valid_coords(walls, self.typical_agent_r)
        gold = self.random_gold(walls, max_num_gold=1, max_agent_offset=max_agent_offset, agent_x=agent_x, agent_y=agent_y)
        while self.spot_overlap_check(agent_x, agent_y, gold[0][0], gold[0][1], self.typical_gold_r, self.typical_agent_r):
            gold = self.random_gold(walls, max_num_gold=1, max_agent_offset=max_agent_offset, agent_x=agent_x, agent_y=agent_y)
        direction = random.uniform(0, 2*math.pi)
        res = Settings(gameSize=gameSize,
#                       indicator_length = self.typical_indicator_length,
                       agent_r = self.typical_agent_r,
                       gold_r = self.typical_gold_r,
                       walls = walls,
                       gold = gold,
                       agent_x = agent_x,
                       agent_y = agent_y,
                       direction = direction)
        return res

    # only use this with only 1 gold on the scene, and for 'bare' games
    def bare_draw_arrow_at_gold(self):
        coin = self.settings.gold[0]
        gx, gy = coin
        ax = self.settings.agent_x
        ay = self.settings.agent_y
        # direction from agent to the gold piece
        pointing = self.direction_angle(ax, ay, gx, gy)
        # delta x, y
        dx = gx - ax
        dy = gy - ay

        distance = math.sqrt(dx*dx + dy*dy)
        length = distance - 1.00 * (self.settings.agent_r + self.settings.gold_r)
        extension = length / (2 * self.settings.agent_r)

        self.draw_arrow(extension, pointing)
        # This will be very useful for the 'face' trick, but that comes later.
        return pointing








