#!/usr/bin/python

'''
Created on Nov 9, 2015

@author: aviadc
'''

#===============================================================================
# Imports
#===============================================================================
import argparse
import sys
import traceback
import pygame
import random
import struct
import os
import datetime
import subprocess
import multiprocessing
import numpy as np
import errno
from objects.target import Target
from objects.host import Host
from utils.exceptions import GameOverException, GameResetException, InvalidRewardSystem
from collections import namedtuple
from configobj import ConfigObj
from utils import services
from utils.consts import SCREEN_WIDTH, SCREEN_HEIGHT, LANE_WIDTH, DT, KPH2MPS, \
    HOST_W, METER2PIX, PIX2METER, HALF_WIDTH, HALF_HEIGHT, R,  NUMBER_OF_TARGETS

from ER import ER
import common
import numpy.matlib

#===============================================================================
# Globals
#===============================================================================
GUI_MODE = True
DEFAULT_REWARD = 'basic_reward'
PSEUDO_RANDOM_SEED = random.randint(1, int((1<<31)-1)) #1 to 32b max int

#===============================================================================
# Consts
#===============================================================================
EXPORT_MOVIE_COMMAND = '/usr/bin/ffmpeg -r 29.9 -i {pics_dir}/image%09d.png -b 10000000 {outpath}'

#===============================================================================
# Simulator
#===============================================================================
class Simulator(object):
    def __init__(self, args):
        self._args = args
        self.size = self.width, self.height = SCREEN_WIDTH, SCREEN_HEIGHT
        self._config = ConfigObj(args.config)
        self._screen = None
        self._running = False
        self._profiles = self._load_profiles(self._config)
        self._pipes = self._open_pipes()
        self._game_round = 0        
        self._wins = 0
        self._loop_count = 0
        self._time = 0.0
        self._t0 = 0.0
        self._fps = args.fps
        
        self._video_image = 0
        self._outpath = None

        self._host_object = None
        self._target_objects = []
        
        self._init_gui()


        self.build_er_arrays()

        self.er_expert = ER(memory_size=200000,
                            state_dim=17,
                            action_dim=1,
                            reward_dim=5,
                            batch_size=32)

        if args.export: self._setup_movie_folder(args.export)
    
    def _init_gui(self):
        pygame.init()
        
        self._build_targets()
        
        if GUI_MODE:
            self._screen = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self._running = True

    def get_game_round(self):
        return self._game_round
    
    def get_loop_count(self):
        return self._loop_count
    
    def get_host(self):
        return self._host_object
    
    def screen(self):
        return self._screen
    
    def get_state_grid(self):
        num_targets = len(self._target_objects)
        target_state = np.zeros((num_targets,4))
        for i, target in enumerate(self._target_objects):
            target_state[i, 0] = target.get_speed()
            target_state[i, 1] = target.get_relx()
            target_state[i, 2] = target.get_X()
            target_state[i, 3] = target.get_acceleration()
        if self._host_object:
            host_state = [self._host_object.get_speed(), self._host_object.get_relx(), self._host_object.get_X()]
        else:
            host_state = [-999,-999,-999]
        return target_state,host_state
    
    def get_targets(self):
        return self._target_objects
    
    def run(self):
        clock = pygame.time.Clock()
        
        gamereset = False
        gameover = False
        
        print 'Game round: {}'.format(self._game_round)

        while self._running:
            try:
                self._event_handler()   
                self._on_loop()
                if GUI_MODE: 
                    a=1
                    self._on_render()
            except GameOverException:
                self._wins += self._host_object.is_win()
                line_info = "GAME OVER after %f seconds" % (self._time-self._t0) 
                print line_info
                #pygame.time.delay(500)
                gameover = True
            except GameResetException:
                gamereset = True
            else:
                self._time += DT
            finally:
                # send game over
                if self._host_object:
                    buf = struct.pack('f',1.0*gameover + 2.0*gamereset)
                    os.write(self._pipes.gameover_pipe,buf)

            #Reset game if required
            if gameover or gamereset or self._loop_count == 65:
                self._reset()
            
            #Turn of game reset \ game over
            gamereset = False
            gameover = False
                    
            # Make sure game doesn't run at more than 60 frames per second
            clock.tick(self._fps)
            self._loop_count += 1
            # if self._loop_count == 65:
            #     self._reset()
            
        self._on_cleanup()
    
    def _event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_LCTRL:
                    self._running = False
                elif event.key == pygame.K_F5:
                    raise GameResetException()
                elif event.key == pygame.K_KP_PLUS:
                    self._fps *= 2
                elif event.key == pygame.K_KP_MINUS:
                    self._fps = max(1,self._fps/2)
    
    def _reset(self):
        self._time = 0
        self._loop_count = 0


        # clip first couple of steps
        self.states = self.states[14:]
        self.actions = self.actions[14:]

        approaching = self.states[:,1] < 0

        self.states = self.states[approaching][:]
        self.actions = self.actions[approaching]

        is_aggressive = np.matlib.repmat(self.is_aggressive, self.states.shape[0], 1)

        terminals = np.zeros(self.states.shape[0])
        terminals[-1] = 1

        # push to er
        self.er_expert.add(actions=self.actions,
                           states=self.states,
                           rewards=is_aggressive,
                           terminals=terminals)

        if self. _game_round == 3000:
            common.save_er(module=self.er_expert,
                           directory='/homes/nirb/work/git/Buffe/Applications/m-GAIL/environments/roundabout')
            print 'saved expert ER'
            sys.exit()
        # reintialize er arrays
        self.build_er_arrays()


        self._game_round += 1 #Set another game
        
        print 'Game round: {}'.format(self._game_round)
        
        self._host_object = None
        self._target_objects = []
        self._build_targets()

    def _build_targets(self):
        #=======================================================================
        # Create Targets
        #=======================================================================
        num_targets = NUMBER_OF_TARGETS#int(self._config.get('settings').get('targets',NUMBER_OF_TARGETS))
        pi = np.pi
        N = num_targets
        P = np.arange(0,(2*pi*R-1e-10), 2*pi*R/N) + (np.random.rand(N)-0.5)*(2*pi*R/N)/2
        relX = np.hstack((np.diff(P),P[0]+2*pi*R-P[N-1]))
        V = 10 + np.zeros(relX.size);
        X = P + R*pi/2 ; X[X > pi*R] = X[X > pi*R]-2*pi*R
        #print 'X ', X
        #print 'relX ', relX         
        self.is_aggressive = np.asarray([0]*num_targets)

        for i in xrange(num_targets):
            availiable_profiles = self._profiles.keys()
            profile_name = random.choice(availiable_profiles)
            profile_obj = self._build_profile_object(profile_name)
            target = Target(self, i, profile_obj, relX[i], X[i])
            self._target_objects.append(target)
            self.is_aggressive[i] = target._is_aggressive
#                print "added target at loop count %d" % self._loop_count

    def build_er_arrays(self):
        self.x_h = np.asarray([0])
        self.v_h = np.asarray([0])
        self.v_t = np.asarray([[0]*5])
        self.states = np.asarray([[0]*17])
        self.actions = np.asarray([0])
        self.is_aggressive = np.asarray([[0]*5])

    def _on_loop(self):
        if GUI_MODE:
            self._screen.fill((1,166,17))

            r = 10-HOST_W/2*PIX2METER ; img_r = -r*METER2PIX
            TWO_LANE_WIDTH = 2 * HOST_W * LANE_WIDTH
            pygame.draw.rect(self._screen, (100,100,100), (HALF_WIDTH - TWO_LANE_WIDTH/2,0, 2 * HOST_W * LANE_WIDTH, SCREEN_HEIGHT))
            pygame.draw.rect(self._screen, (100,100,100), (0, HALF_HEIGHT - TWO_LANE_WIDTH/2, SCREEN_WIDTH, 2 * HOST_W * LANE_WIDTH))
            pygame.draw.circle(self._screen,(100,100,100), (HALF_WIDTH,HALF_HEIGHT), int(1.1*img_r))
            pygame.draw.circle(self._screen,(155,155,155), (HALF_WIDTH,HALF_HEIGHT), int(img_r*0.6))

        #=======================================================================
        # Create Host
        #=======================================================================
        if self._loop_count == 10 and not self._host_object:
            reward_obj = self._build_reward_system(self._args.reward)
            host_speed = float(self._config.get('settings').get('host_speed')) * KPH2MPS
            self._host_object = Host(self, reward_obj, speed=host_speed)

        state = self.get_state_grid()

        x_h = state[1][2]
        v_h = (x_h  - self.x_h[-1])/DT
        v_t = state[0][:, 0]
        x_t = state[0][:, 2]
        a_t = state[0][:, 3]#  v_t - self.v_t[-1]
        action = v_h - self.v_h[-1]
        state_i = np.concatenate([[v_h], [x_h], v_t, x_t, a_t])

        self.v_h = np.append(self.v_h, [v_h], axis=0)
        self.x_h = np.append(self.x_h, [x_h], axis=0)
        self.v_t = np.append(self.v_t, [v_t], axis=0)

        self.states = np.append(self.states, [state_i], axis=0)
        self.actions = np.append(self.actions, [action], axis=0)

        # print v_h

        #This will update Host Commands and State
        if self._host_object:
            self._host_object.move(state,self._pipes)
    
        #Move targets
        for target in self._target_objects:
            target.move(self._time, state)
        
    def _on_render(self):
        # Draw Targets
        for t in self._target_objects:
            t.draw()

        # Draw Host
        if self._host_object:
            self._host_object.draw()
        
        # Draw game info
        font = pygame.font.Font(None, 20)
        info = 'Wins: {} out of {}, round {}'.format(self._wins,self._game_round,self._loop_count)
        scoretext = font.render('{} '.format(info), True, (255, 255, 255),(0,0,0))
        self._screen.blit(scoretext, (0,SCREEN_HEIGHT-20))
        
        # Update Screen
        pygame.display.update()
        
        #Save image for video if needed
        if self._outpath:
            pygame.image.save(self._screen, self._outpath + "/image{round}.png".format(round=str(self._video_image).zfill(9)))
            self._video_image += 1
    
    def _on_cleanup(self):
        pygame.quit()
        if self._outpath: self._export_movie(self._outpath)
    
    def _load_profiles(self, config_obj):
        profiles = {}
        
        for profile_name, profile_attrs in config_obj.get('profiles',{}).iteritems():
            print 'loading profile: {}'.format(profile_name),
            try:
                qualified_name = 'profiles.' + profile_name
                __import__(qualified_name)
                mod = sys.modules[qualified_name]
                profile_cls = getattr(mod, profile_name)
                if profile_cls:
                    profiles.setdefault(profile_name,{'cls':profile_cls,'attrs':profile_attrs})
            except:
                traceback.print_exc()
                print 'Failed.'
            else: print 'Success'
        return profiles

    def _build_profile_object(self, profile_name):
        profile_cls = self._profiles[profile_name]['cls']
        profile_attrs = self._profiles[profile_name]['attrs']
        profile_obj = profile_cls(**profile_attrs)
        return profile_obj
    
    def _build_reward_system(self, reward_name):
        reward_cls = services.load_module('rewards', reward_name)
    
        if not reward_cls:
            #traceback.print_exc()
            e = 'Unable to find reward module named {}. Exiting.'.format(reward_name)
            raise InvalidRewardSystem(e)
    
        reward_obj = reward_cls()
        return reward_obj
        
    def _open_pipes(self):
        #return None
    
        PipesTuple = namedtuple('pipe_t', ['state_pipe', 'reward_pipe', 'gameover_pipe', 'action_pipe'])
        
        pipes = PipesTuple(state_pipe    = os.open("/tmp/piped_state.bin", os.O_WRONLY),
                           reward_pipe   = os.open("/tmp/piped_reward.bin", os.O_WRONLY),
                           gameover_pipe = os.open("/tmp/piped_gameOver.bin", os.O_WRONLY),
                           action_pipe   = os.open("/tmp/piped_action.bin", os.O_RDONLY))
        
        return pipes
    
    def _setup_movie_folder(self, outdir):
        dt = datetime.datetime.now().strftime("%d%m%y%H%M")
        outpath = os.path.join(os.path.abspath(outdir), 'shtutit_{}'.format(dt))
        print 'Exporting to: {}'.format(outpath)
        os.makedirs(outpath)
        self._outpath = outpath
    
    def _export_movie(self,outpath):
        movie_path = "{}.mp4".format(outpath)            
        export_movie_command = EXPORT_MOVIE_COMMAND.format(pics_dir=outpath, outpath=movie_path)
        print "Running:\n",export_movie_command
        p = subprocess.Popen(export_movie_command, shell=True)
        stdoutdata, stderrdata = p.communicate()
        print '\nMovie created at: {}'.format(movie_path)
    
#===============================================================================
# Main
#===============================================================================
def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",'--config', default='config.ini', help='Profiles config')
    parser.add_argument("-p","--policy", default=None, help='Specify a policy game name')
    parser.add_argument("-r","--reward", default=DEFAULT_REWARD, help='Specify rewarding system')
    parser.add_argument("-s","--seed", type=int, default=None, help='Specify a pseudo random seed')
    parser.add_argument("--fps",'--time', type=float, default=60, help='Set time sleep')
    parser.add_argument("--export", default=None, help='Export movie')
    parser.add_argument("--nogui", action='store_true', help='Hide GUI')

    args = parser.parse_args()
    return args

def _run_demo_policy(policy_name):
    policy_cls = services.load_module('policies', policy_name)
    
    if not policy_cls:
        #traceback.print_exc()
        print 'Unable to find policy module.'
        print 'Waiting for a policy...'
        return policy_cls

    policy_object = policy_cls()
    p = multiprocessing.Process(target=policy_object.run)
    p.start()
    return p
    
def _create_pipes(path='/tmp/'):
    pipes = ["piped_state.bin", "piped_reward.bin", "piped_gameOver.bin", "piped_action.bin"]
    for p in pipes:
        try: os.mkfifo(path + p)
        except OSError as e:
            if e.errno == errno.EEXIST: pass
            else: raise
    
def main(args):
    global GUI_MODE, PSEUDO_RANDOM_SEED
    GUI_MODE = not args.nogui
    
    # Handle seed
    if args.seed is not None:
        PSEUDO_RANDOM_SEED = args.seed
    random.seed(PSEUDO_RANDOM_SEED)
    np.random.seed(seed=PSEUDO_RANDOM_SEED)
    print 'Pseudo random seed number: {}'.format(PSEUDO_RANDOM_SEED)
    
    # Create Pipes
    _create_pipes()
    
    # Handle demo policy
    policy = None
    if args.policy is not None:
        policy = _run_demo_policy(args.policy)
    else:
        print 'Waiting for a policy...'
    
    # Run simulator
    try: simulator = Simulator(args)
    except InvalidRewardSystem as e: print e
    else: simulator.run()
    
    if policy: policy.terminate()

#------------------------------------------------------------------------------ 
if __name__ == '__main__':
    args = get_command_line_args()
    main(args)

    
    
    