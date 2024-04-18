import pygame as pg
import numpy as np
import time
import scipy.io
import os
from examples.choose_problem_intersection import system, problem, config

LIGHT_GREY = (230, 230, 230)

class VisUtils:

    def __init__(self):
        # change this name:
        # self.model_name = 'Empathetic'
        self.model_name = 'Non-Empathetic'
        self.screen_width = 10  # 50
        self.screen_height = 10  # 50
        self.coordinate_scale = 80
        self.zoom = 0.25
        self.asset_location = 'assets/'
        self.fps = 24  # max framework

        self.car_width = problem.W1
        self.car_length = problem.L1
        self.road_length = problem.R1 / 2.

        load_path = 'examples/vehicle/data_E_a_a_belief_na_na.mat'
        # load_path = 'examples/vehicle/data_NE_a_a_belief_na_na.mat'
        # load_path = 'examples/vehicle/data_E_na_na_belief_a_a.mat'
        # load_path = 'examples/vehicle/data_NE_na_na_belief_a_a.mat'
        self.train_data = scipy.io.loadmat(load_path)

        self.new_data = self.train_data

        self.T = self.new_data['t']

        self.car_par = [{'sprite': 'red_car_sized.png',
                         'state': self.new_data['X'][:1, :],
                         'policy': self.new_data['P'][:1, :],
                         'orientation': 0.},
                        {'sprite': 'blue_car_sized.png',
                         'state': self.new_data['X'][1:, :],
                         'policy': self.new_data['P'][1:, :],
                         'orientation': 90.}
                        ]

        img_width = int(self.car_width * self.coordinate_scale * self.zoom)
        img_height = int(self.car_length * self.coordinate_scale * self.zoom)

        "initialize pygame"
        pg.init()
        self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                           self.screen_height * self.coordinate_scale))

        self.car_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location + self.car_par[i]['sprite']),
                                               (img_width, img_height)), - self.car_par[i]['orientation']) for i in range(len(self.car_par))]

        self.origin = np.array([35, 35])

        "Draw Axis Lines"

        self.screen.fill((255, 255, 255))
        self.draw_axes() #calling draw axis function
        pg.display.flip()
        pg.display.update()

    def draw_frame(self):
        '''state[t] = [s_x, s_y, v_x, v_y]_t'''
        '''state = [state_t, state_t+1, ...]'''
        # Draw the current frame
        '''frame is counting which solution step'''

        # steps = self.T.shape[0]  # 10/0.1 + 1 = 101
        steps = self.T.shape[1]

        for k in range(steps - 1):
            self.screen.fill((255, 255, 255))
            self.draw_axes()
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
                '''transform pos'''
                pixel_pos_car = self.c2p(pos)
                size_car = self.car_image[i].get_size()
                self.screen.blit(self.car_image[i],
                                 (pixel_pos_car[0] - size_car[0] / 2, pixel_pos_car[1] - size_car[1] / 2))
                time.sleep(0.05)
                # if self.sim.decision_type == "baseline":
                #     time.sleep(0.05)
            # Annotations
            font = pg.font.SysFont("Times New Roman", 40)  # 25
            screen_w, screen_h = self.screen.get_size()
            label_x = screen_w - 720
            label_y = 50
            label = font.render("Model Type: {}".format(self.model_name), 1, (0, 0, 0))
            self.screen.blit(label, (label_x, label_y))

            label_y += 40
            label_policy_1 = font.render("P1 policy: {}".format(self.car_par[0]['policy'][0][k+1]), 1, (0, 0, 0))
            self.screen.blit(label_policy_1, (label_x, label_y))
            label_y += 40
            label_policy_2 = font.render("P2 policy: {}".format(self.car_par[1]['policy'][0][k+1]), 1, (0, 0, 0))
            self.screen.blit(label_policy_2, (label_x, label_y))

            # label_y_offset = 30
            # pos_h, speed_h = self.car_par[0]['state'][0][k+1], self.car_par[0]['state'][1][k+1]  #y axis
            # label = font.render("Car 1 position and speed: (%5.4f , %5.4f)" % (pos_h, speed_h), 1,
            #                     (0, 0, 0))
            # self.screen.blit(label, (label_x, label_y))
            # pos_m, speed_m = self.car_par[1]['state'][0][k+1], self.car_par[1]['state'][1][k+1] #x axis
            # label = font.render("Car 2 position and speed: (%5.4f , %5.4f)" % (pos_m, speed_m), 1,
            #                     (0, 0, 0))
            # self.screen.blit(label, (label_x, label_y + label_y_offset))
            #
            # label = font.render("Frame: %i" % steps, 1, (0, 0, 0))
            # self.screen.blit(label, (10, 10))

            recording_path = 'image_recording/'
            pg.image.save(self.screen, "%simg%03d.png" % (recording_path, k))

            "drawing the map of state distribution"
            # pg.draw.circle(self.screen, (255, 255, 255), self.c2p(self.origin), 10)  # surface,  color, (x, y),radius>=1

            # time.sleep(1)

            pg.display.flip()
            pg.display.update()

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
        t_bar = np.arange(0, 1.3, 0.05)
        X_bar = np.zeros((2, t_bar.shape[0]))
        P_bar = np.zeros((2, t_bar.shape[0]))
        i = 0
        j = 0
        time = 0
        t = data['t']  # time is from train_data
        X = data['X']
        P = data['P']
        while time <= 1.2:
            while t[0][i] <= time:
                i += 1
            """
            agent 1: original state: (py, px)
            agent 2: original state: (px, py)
            """
            X_bar[0][j] = (time - t[0][i - 1]) * (X[0][i] - X[0][i - 1]) / (t[0][i] - t[0][i - 1]) + X[0][i - 1]
            X_bar[1][j] = (time - t[0][i - 1]) * (X[1][i] - X[1][i - 1]) / (t[0][i] - t[0][i - 1]) + X[1][i - 1]
            P_bar[0][j] = (time - t[0][i - 1]) * (P[0][i] - P[0][i - 1]) / (t[0][i] - t[0][i - 1]) + P[0][i - 1]
            P_bar[1][j] = (time - t[0][i - 1]) * (P[1][i] - P[1][i - 1]) / (t[0][i] - t[0][i - 1]) + P[1][i - 1]

            time = time + 0.1
            j += 1

        new_data = dict()
        new_data.update({'t': t_bar,
                         'X': X_bar,
                         'P': P_bar})

        return new_data

if __name__ == '__main__':
    vis = VisUtils()
    vis.draw_frame()

    path = 'image_recording/'
    import glob
    image = glob.glob(path+"*.png")
    # print(image)
    # episode_step_count = len(image)
    img_list = image # [path + "img" + str(i).zfill(3) + ".png" for i in range(episode_step_count)]

    import imageio
    images = []
    for filename in img_list:
        images.append(imageio.imread(filename))
    # tag = 'E' + '_' + 'theta1' + '=' + 'a' + '_' + 'theta2' + '=' + 'a' + '_' + 'time horizon' + '=' + str(config.t1)
    tag = 'NE' + '_' + 'theta1' + '=' + 'a' + '_' + 'theta2' + '=' + 'a' + '_' + 'time horizon' + '=' + str(config.t1)
    # tag = 'E' + '_' + 'theta1' + '=' + 'na' + '_' + 'theta2' + '=' + 'na' + '_' + 'time horizon' + '=' + str(config.t1)
    # tag = 'NE' + '_' + 'theta1' + '=' + 'na' + '_' + 'theta2' + '=' + 'na' + '_' + 'time horizon' + '=' + str(config.t1)
    imageio.mimsave(path + 'movie_' + tag + '.gif', images, 'GIF', duration=0.12)
    # Delete images
    [os.remove(path + file) for file in os.listdir(path) if ".png" in file]

