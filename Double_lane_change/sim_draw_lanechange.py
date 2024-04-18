import pygame as pg
import numpy as np
import time
import scipy.io
import os
from examples.choose_problem_lane_change import system, problem, config
import math

LIGHT_GREY = (230, 230, 230)

class VisUtils:

    def __init__(self):
        # change this name:
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

        load_path = 'examples/vehicle/data_lane_GT.mat'
        # load_path = 'examples/vehicle/data_lane_SSL.mat'
        # load_path = 'examples/vehicle/data_lane_SL.mat'
        # load_path = 'examples/vehicle/data_lane_HL.mat'
        # load_path = 'examples/vehicle/data_lane_VH.mat'
        # load_path = 'examples/vehicle/data_lane_EL.mat'

        self.train_data = scipy.io.loadmat(load_path)

        self.new_data = self.generate(self.train_data)

        self.T = self.new_data['t']

        self.car_par = [{'sprite': 'blue_car_sized.png',
                         'state': self.new_data['X'][:2, :],  # pos_x, vel_x
                         'orientation': -90,
                         'angular': self.new_data['Omega'][:1, :]},
                        {'sprite': 'red_car_sized.png',
                         'state': self.new_data['X'][2:, :],  # pos_x, vel_x
                         'orientation': 90,
                         'angular': self.new_data['Omega'][1:, :]}
                        ]

        img_width = int(self.car_width * self.coordinate_scale * self.zoom)
        img_height = int(self.car_length * self.coordinate_scale * self.zoom)

        "initialize pygame"
        pg.init()
        self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                           self.screen_height * self.coordinate_scale))

        self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location + self.car_par[i]['sprite']),
                                               (img_width, img_height)), - self.car_par[i]['orientation']) for i in range(len(self.car_par))]

        self.coor_image = pg.image.load(self.asset_location + self.coordinate)

        # we can change the number to adjust the position of the road frame
        self.origin = np.array([35, 35])  # 35, 35; 30, 30; 36, 36

        # self.origin = np.array([0, 0])

        "Draw Axis Lines"

        self.screen.fill((255, 255, 255))
        self.draw_axes() # calling draw axis function
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

        img_width = int(self.car_width * self.coordinate_scale * self.zoom)
        img_height = int(self.car_length * self.coordinate_scale * self.zoom)

        for k in range(steps - 1):
            self.screen.fill((134, 189, 119))  # grass color: 134, 189, 119; white: 255, 255, 255
            self.draw_axes()
            # self.draw_dashed_line1()
            self.draw_dashed_line2()
            # Draw Images
            n_agents = 2
            for i in range(n_agents):
                '''getting pos of agent: (x, y)'''
                pos_x_old = np.array(self.car_par[i]['state'][0][k])  # car x position
                pos_x_new = np.array(self.car_par[i]['state'][0][k + 1])  # get 0 and 1 element (not include 2) : (x, y)

                pos_y_old = np.array(self.car_par[i]['state'][1][k])  # car y position
                pos_y_new = np.array(self.car_par[i]['state'][1][k + 1])  # get 0 and 1 element (not include 2) : (x, y)
                '''smooth out the movement between each step'''
                pos_x = pos_x_old * (1 - k * 1. / steps) + pos_x_new * (k * 1. / steps)
                pos_y = pos_y_old * (1 - k * 1. / steps) + pos_y_new * (k * 1. / steps)

                if i == 0:
                    pos = (82 - pos_x, pos_y)  # car position
                    orientation = 90 - np.array(self.car_par[i]['angular'][0][k]) * 180 / math.pi

                if i == 1:
                    pos = (82 - pos_x, pos_y)  # car position
                    orientation = 90 - np.array(self.car_par[i]['angular'][0][k]) * 180 / math.pi

                '''transform pos'''
                pixel_pos_car = self.c2p(pos)
                car_image = pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location + self.car_par[i]['sprite']),
                                       (img_width, img_height)), -orientation)

                size_car = car_image.get_size()

                self.screen.blit(car_image, (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))
                time.sleep(0.05)
           
            recording_path = 'image_recording/'
            pg.image.save(self.screen, "%simg%03d.png" % (recording_path, k))

            "drawing the map of state distribution"
            # pg.draw.circle(self.screen, (255, 255, 255), self.c2p(self.origin), 10)  # surface,  color, (x, y),radius>=1

            # time.sleep(1)

            pg.display.flip()
            pg.display.update()

    def draw_axes(self):
        # draw lanes based on environment
        # pg.draw.line(self.screen, LIGHT_GREY, self.c2p((35, -50)),
        #              self.c2p((35, 100)), self.car_image[0].get_size()[1] * 4)

        # for uncontrolled intersection case, set self.car_image[1].get_size()[1]
        # for unprotected left turn, set self.car_image[1].get_size()[0]

        pg.draw.line(self.screen, LIGHT_GREY, self.c2p((100, 35)),
                     self.c2p((-50, 35)), self.car_image[1].get_size()[1] * 5)  #LIGHT_GREY


    def draw_dashed_line1(self):
        # origin = self.c2p((35 + self.car_width, -50))
        # target = self.c2p((35 + self.car_width, 100))

        origin = self.c2p((35, -50))
        target = self.c2p((35, 100))
        displacement = target - origin
        length = abs(displacement[1])
        slope = displacement / length
        dash_length = 10

        for index in range(0, int(length / dash_length), 2):
            start = origin + (slope * index * dash_length)
            end = origin + (slope * (index + 1) * dash_length)
            pg.draw.line(self.screen, (0, 0, 0), start, end, 1)

    def draw_dashed_line2(self):
        # origin = self.c2p((100, 35 - self.car_width))
        # target = self.c2p((-50, 35 - self.car_width))

        origin = self.c2p((100, 35))
        target = self.c2p((-50, 35))
        displacement = target - origin
        length = abs(displacement[0])
        slope = displacement / length
        dash_length = 10

        for index in range(0, int(length / dash_length), 2):
            start = origin + (slope * index * dash_length)
            end = origin + (slope * (index + 1) * dash_length)
            pg.draw.line(self.screen, (0, 0, 0), start, end, 1)

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
        t_bar = np.linspace(0, 4, 26)
        X_bar = np.zeros((4, t_bar.shape[0]))
        Omega_bar = np.zeros((2, t_bar.shape[0]))
        i = 0
        j = 0
        time = 0
        t = data['t']  # time is from train_data
        X = data['X']
        while time <= 4.0:
            while t[0][i] <= time:
                i += 1
            """
            agent 1: original state: (py, px)
            agent 2: original state: (px, py)
            """
            X_bar[0][j] = (time - t[0][i - 1]) * (X[0][i] - X[0][i - 1]) / (t[0][i] - t[0][i - 1]) + X[0][i - 1]
            X_bar[1][j] = (time - t[0][i - 1]) * (X[1][i] - X[1][i - 1]) / (t[0][i] - t[0][i - 1]) + X[1][i - 1]
            X_bar[2][j] = (time - t[0][i - 1]) * (X[4][i] - X[4][i - 1]) / (t[0][i] - t[0][i - 1]) + X[4][i - 1]
            X_bar[3][j] = (time - t[0][i - 1]) * (X[5][i] - X[5][i - 1]) / (t[0][i] - t[0][i - 1]) + X[5][i - 1]

            Omega_bar[0][j] = (time - t[0][i - 1]) * (X[2][i] - X[2][i - 1]) / (t[0][i] - t[0][i - 1]) + X[2][i - 1]
            Omega_bar[1][j] = (time - t[0][i - 1]) * (X[6][i] - X[6][i - 1]) / (t[0][i] - t[0][i - 1]) + X[6][i - 1]
            time = time + 0.16
            j += 1

        # only for SSL
        # for i in range(26-16):
        #     X_bar[0][i + 16] = X_bar[0][15]
        #     X_bar[1][i + 16] = X_bar[1][15]
        #     X_bar[2][i + 16] = X_bar[2][15]
        #     X_bar[3][i + 16] = X_bar[3][15]
        #     Omega_bar[0][i + 16] = Omega_bar[0][15]
        #     Omega_bar[1][i + 16] = Omega_bar[1][15]

        # only for SL
        # for i in range(26-20):
        #     X_bar[0][i + 20] = X_bar[0][19]
        #     X_bar[1][i + 20] = X_bar[1][19]
        #     X_bar[2][i + 20] = X_bar[2][19]
        #     X_bar[3][i + 20] = X_bar[3][19]
        #     Omega_bar[0][i + 20] = Omega_bar[0][19]
        #     Omega_bar[1][i + 20] = Omega_bar[1][19]

        # only for VH
        # for i in range(26-10):
        #     X_bar[0][i + 10] = X_bar[0][9]
        #     X_bar[1][i + 10] = X_bar[1][9]
        #     X_bar[2][i + 10] = X_bar[2][9]
        #     X_bar[3][i + 10] = X_bar[3][9]
        #     Omega_bar[0][i + 10] = Omega_bar[0][9]
        #     Omega_bar[1][i + 10] = Omega_bar[1][9]

        # only for HJ PDE
        # for i in range(26-7):
        #     X_bar[0][i + 7] = X_bar[0][8]
        #     X_bar[1][i + 7] = X_bar[1][8]
        #     X_bar[2][i + 7] = X_bar[2][8]
        #     X_bar[3][i + 7] = X_bar[3][8]
        #     Omega_bar[0][i + 7] = Omega_bar[0][8]
        #     Omega_bar[1][i + 7] = Omega_bar[1][8]

        # only for GT, HL
        X_bar[0][-1] = X[0][-1]
        X_bar[1][-1] = X[1][-1]
        X_bar[2][-1] = X[4][-1]
        X_bar[3][-1] = X[5][-1]
        Omega_bar[0][-1] = X[2][-1]
        Omega_bar[1][-1] = X[6][-1]

        new_data = dict()
        new_data.update({'t': t_bar,
                         'X': X_bar,
                         'Omega': Omega_bar})

        return new_data


if __name__ == '__main__':
    vis = VisUtils()
    vis.draw_frame()

    path = 'image_recording/'
    import glob

    image = glob.glob(path + "*.png")
    img_list = image  # [path + "img" + str(i).zfill(3) + ".png" for i in range(episode_step_count)]

    import imageio

    images = []
    for filename in img_list:
        images.append(imageio.imread(filename))
    tag = 'case2_simulation'
    imageio.mimsave(path + 'movie_' + tag + '.gif', images, 'GIF', fps=9)  # fps=7
    # imageio.mimsave(path + 'movie_' + tag + '.gif', images, 'GIF', duration=0.2)
    # Delete images
    [os.remove(path + file) for file in os.listdir(path) if ".png" in file]
