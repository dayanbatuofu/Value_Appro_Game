import pygame as pg
import numpy as np
import time
import scipy.io
import os
from examples.choose_problem_intersection import system, problem, config
import math

LIGHT_GREY = (230, 230, 230)

class VisUtils:

    def __init__(self):
        # change this name:
        self.model_name = 'PMP'
        self.screen_width = 10  # 50
        self.screen_height = 10  # 50
        self.coordinate_scale = 80
        self.zoom = 0.25  # 0.25 change the number to adjust the position of the road frame
        self.asset_location = 'assets/'
        self.fps = 24  # max framework

        self.car_width = problem.W1
        self.car_length = problem.L1
        self.road_length = problem.R1 / 2.
        self.coordinate = 'coordinates.png'

        load_path = 'examples/vehicle/data_a_a_GT.mat'
        # load_path = 'examples/vehicle/data_a_a_PINN.mat'
        # load_path = 'examples/vehicle/data_a_a_SL.mat'
        # load_path = 'examples/vehicle/data_a_a_HL.mat'
        # load_path = 'examples/vehicle/data_a_a_VH.mat'
        # load_path = 'examples/vehicle/data_a_a_EL.mat'

        self.train_data = scipy.io.loadmat(load_path)

        self.new_data = self.generate(self.train_data)

        self.T = self.new_data['t']

        self.car_par = [{'sprite': 'red_car_sized.png',
                         'state': self.new_data['X'][:1, :],  # pos_y, pos_x
                         'orientation': 0.},
                        {'sprite': 'blue_car_sized.png',
                         'state': self.new_data['X'][1:, :],  # pos_x, pos_y
                         'orientation': 90.}
                        ]

        img_width = int(self.car_width * self.coordinate_scale * self.zoom)
        img_height = int(self.car_length * self.coordinate_scale * self.zoom)

        "initialize pygame"
        pg.init()
        self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                           self.screen_height * self.coordinate_scale))

        self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location + self.car_par[i]['sprite']),
                                               (img_width, img_height)), -self.car_par[i]['orientation']) for i in range(len(self.car_par))]

        self.coor_image = pg.image.load(self.asset_location + self.coordinate)

        # we can change the number to adjust the position of the road frame
        self.origin = np.array([35, 35])  # 35, 35; 30, 30;

        # self.origin = np.array([0, 0])

        "Draw Axis Lines"

        self.screen.fill((255, 255, 255))
        self.draw_axes()  # calling draw axis function
        pg.display.flip()
        pg.display.update()

    def blit_alpha(self, target, source, location, opacity):
        x = location[0]
        y = location[1]
        temp = pg.Surface((source.get_width(), source.get_height())).convert()
        temp.blit(target, (-x, -y))
        temp.blit(source, (0, 0))
        temp.set_alpha(opacity)
        target.blit(temp, location)

    def draw_frame(self):
        '''state[t] = [s_x, s_y, v_x, v_y]_t'''
        '''state = [state_t, state_t+1, ...]'''
        # Draw the current frame
        '''frame is counting which solution step'''

        steps = self.T.shape[0]  # 10/0.1 + 1 = 101
        # steps = self.T.shape[1]

        self.screen.fill((255, 255, 255))
        self.draw_axes()

        img_width = int(self.car_width * self.coordinate_scale * self.zoom)
        img_height = int(self.car_length * self.coordinate_scale * self.zoom)

        for k in range(steps - 1):
            # self.screen.fill((255, 255, 255))
            # self.draw_axes()
            # Draw Images
            n_agents = 2
            for i in range(n_agents):
                '''getting pos of agent: (x, y)'''
                pos_old = np.array(self.car_par[i]['state'][0][k])  # car position
                pos_new = np.array(self.car_par[i]['state'][0][k + 1])  # get 0 and 1 element (not include 2) : (x, y)

                '''smooth out the movement between each step'''
                pos = pos_old * (1 - k * 1. / steps) + pos_new * (k * 1. / steps)
                if i == 0:
                    pos = (self.road_length, pos)  # car position
                if i == 1:
                    pos = (70 - pos, self.road_length)  # car position

                pixel_pos_car = self.c2p(pos)
                size_car = self.car_image[i].get_size()

                # try with opacity
                self.blit_alpha(self.screen, self.car_image[i].convert_alpha(), (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2), 30*k)
                time.sleep(0.05)

            # time.sleep(1)

            pg.display.flip()
            pg.display.update()

        recording_path = 'image_recording/'
        pg.image.save(self.screen, "%simg%03d.png" % (recording_path, k))

    def draw_axes(self):
        # draw lanes based on environment
        pg.draw.line(self.screen, LIGHT_GREY, self.c2p((35, -50)), self.c2p((35, 100)), 35)
        pg.draw.line(self.screen, LIGHT_GREY, self.c2p((100, 35)), self.c2p((-50, 35)), 35)

    def c2p(self, coordinates):
        '''coordinates = x, y position in your environment(vehicle position)'''
        x = self.coordinate_scale * (- coordinates[0] + self.origin[0] + self.screen_width / 2)
        y = self.coordinate_scale * (- coordinates[1] + self.origin[1] + self.screen_height / 2)

        x = int(
            (x - self.screen_width * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_width * self.coordinate_scale * 0.5)
        y = int(
            (y - self.screen_height * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_height * self.coordinate_scale * 0.5)
        '''returns x, y for the pygame window'''

        return np.array([x, y])

    def generate(self, data):
        # t_bar = np.linspace(0, 0.86, 11)  # VH,EL
        t_bar = np.linspace(0, 9, 10)  # SL,SSL,GT,HL
        X_bar = np.zeros((2, t_bar.shape[0]))
        i = 0
        j = 0
        time = 0
        t = data['t']  # time is from train_data
        X = data['X']

        while time <= 0.9:  # VH,EL:0.86; SSL,SL,GT,HL:0.9
            while t[0][i] <= time:
                i += 1
            """
            agent 1: original state: (py, px)
            agent 2: original state: (px, py)
            """
            X_bar[0][j] = (time - t[0][i - 1]) * (X[0][i] - X[0][i - 1]) / (t[0][i] - t[0][i - 1]) + X[0][i - 1]
            X_bar[1][j] = (time - t[0][i - 1]) * (X[2][i] - X[2][i - 1]) / (t[0][i] - t[0][i - 1]) + X[2][i - 1]
            time = time + 0.1  # VH,EL: 0.086; SSL,SL,GT,HL:0.1
            j += 1

        new_data = dict()

        new_data.update({'t': t_bar[:-1],
                         'X': X_bar[:, :-1]})

        return new_data

if __name__ == '__main__':
    vis = VisUtils()
    vis.draw_frame()

