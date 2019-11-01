import numpy as np
import matplotlib
from matplotlib.patches import Polygon as mPolygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import json
from shapely.geometry import Polygon, MultiPolygon, Point


class EnvironmentDef:

    def __init__(self, input_file, factor=5):
        self.plot_obstacles_polygon = []
        self.obs_polygon = MultiPolygon()
        self.initial_state, self.goal_state = [], []
        self.resolution = 0
        self.read_env_from_file(input_file)
        self.boxes = []
        self.factor = factor
        self.convert_to_block()
        # self.print_boxes()

    def read_env_from_file(self, input_file):
        try:
            with open(input_file, mode='r', encoding='utf-8') as a_file:
                environment = json.loads(a_file.read())
        except FileNotFoundError:
            print("File not found")
            exit(1)
        except ValueError:
            print("Invalid JSON")
            exit(1)
        except Exception:
            print("Unable to process input file")
            exit(1)
        try:
            environment['resolution'] and environment['obstacles']
            environment['initial_state'] and environment['goal_state']
        except KeyError:
            print("Invalid Environment definition")
            exit(1)
        self.initial_state, self.goal_state = environment['initial_state'], environment['goal_state']
        self.resolution = environment['resolution']
        temp_polygon_list = []
        for obs in environment['obstacles']:
            if not obs.get('shape') and obs.get('property') and obs['property'].get('vertices'):
                print("Shape element not present for the obstacles")
                continue
            if obs['shape'] == 'polygon':
                # print("Polygon with vertices %s" %(np.array(obs['property']['vertices'])/100))
                polygon = mPolygon(np.array(obs['property']['vertices']))
                temp_polygon_list.append(Polygon(obs['property']['vertices']))
                self.plot_obstacles_polygon.append(polygon)
            else:
                print("Undefined shape")
                break
        self.obs_polygon = MultiPolygon(temp_polygon_list)

    def convert_to_block(self):

        self.boxes = [[1 if self.internal_is_point_inside(i_x/self.factor, i_y/self.factor) else 0
                       for i_x in range(0, self.resolution*self.factor)]
                 for i_y in range(0, self.resolution*self.factor)]
        # self.print_boxes()
        # for polygon in self.obs_polygon:
        #     print(polygon.bounds)

    def print_boxes(self):
        self.boxes.reverse()
        for row in self.boxes:
            for element in row:
                print(element, " ", end="")
            print("\n")
        self.boxes.reverse()

    def internal_is_point_inside(self, x, y):
        return Point(x, y).within(self.obs_polygon)

    def is_point_inside_box(self, y, x):
        """
        :param x:
        :param y:
        :return:True if point is inside, and False if outside
        """
        return False if self.boxes[x*self.factor][y*self.factor] == 0 else True

    def draw_env(self, path):
        fig, ax = plt.subplots()
        x_path, y_path = [], []
        for ls in path:
            x_path.append(ls.position[0])
            y_path.append(ls.position[1])
        colors = 100*np.random.rand(len(self.plot_obstacles_polygon))
        p = PatchCollection(self.plot_obstacles_polygon, cmap=matplotlib.cm.jet, alpha=0.4)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        plt.colorbar(p)
        plt.plot([self.initial_state[0]], [self.initial_state[1]], 'bs', self.goal_state[0], self.goal_state[1], 'g^',
                 x_path, y_path, '*')
        plt.axis([0, self.resolution, 0, self.resolution])
        plt.show()

    def __str__(self):
        return "Obstacle list: %s\nInitial State: %s\nGoal State: %s\nResolution: %d\n" \
               % ([cord.xy for cord in self.plot_obstacles_polygon], self.initial_state, self.goal_state, self.resolution)


