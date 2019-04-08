import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint, choice
import sys

side = 4
size = 512
population_size = 10
mutation_rate = 0.1
s_count = size // side
target = None


def load_image(path):
    main_img = cv2.imread(path)
    main_img = cv2.resize(main_img, (512, 512))
    return main_img


def average_color(image: np.ndarray):
    new = np.zeros((s_count, s_count, 3), np.uint8)
    for i in range(s_count):
        for j in range(s_count):
            a = image[side * i:side * (i + 1), side * j:side * (j + 1)]
            av = np.mean(a, axis=(0, 1))
            new[i][j] = av
    return new


def plot_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def plot_circles(image):
    new_im = np.zeros((size, size, 3), np.uint8)
    new_im[:, :] = np.array([255, 255, 255])
    for i in image.circles:
        center = i[0][1] * side, i[0][0] * side
        cv2.circle(new_im, center, *i[1:])
    plt.imshow(cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB))
    plt.show()


def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)


def rand_radius():
    return randint(20, 21)


def plot_error(x, y):
    plt.xlabel('Generation number')
    plt.ylabel('Best error')
    plt.title("Evolution of population")
    plt.plot(x, y)
    plt.show()


class Individual():

    def __init__(self, circles=None):
        if circles is None:
            self.circles = []
        else:
            self.circles = circles

    @property
    def fitness(self):
        score = 0
        for i in self.circles:
            score += np.sum(abs(i[2] - target[i[0]]))
        return score


class Population:

    @staticmethod
    def generate_random_population():
        return [Population.generate_random_parent() for i in range(population_size)]

    @staticmethod
    def selection(population: list):
        inds = []
        for ind, img in enumerate(population):
            inds.append((img, img.fitness,))
        inds.sort(key=lambda i: i[1])
        inds = inds[:population_size]
        res = []
        for ind_best in inds:
            res.append(ind_best[0])
        return res

    @staticmethod
    def crossover(population: list):
        res = []
        for i in range(population_size * 2):
            i1 = Individual(choice(population).circles.copy())
            i2 = Individual(choice(population).circles.copy())
            start = randint(0, len(i1.circles))
            end = randint(start, len(i1.circles))
            ch_circles = i1.circles[:start] + i2.circles[start:end] + i1.circles[end:]
            res.append(Individual(ch_circles))
        return res

    @staticmethod
    def evolution():
        pop = Population.generate_random_population()
        errors = []
        x = []
        for i in range(100):
            pop = Population.selection(pop)
            pop = Population.crossover(pop)
            for ind in pop:
                Population.mutate(ind)
            if i % 10 == 0:
                errors.append([pop[0].fitness / len(pop[0].circles)])
                x.append(i)
        plot_circles(pop[0])
        return pop[0]

    @staticmethod
    def generate_random_parent():
        parent = Individual()
        for i in range(s_count):
            for j in range(s_count):
                x = i
                y = j
                parent.circles.append(((x, y), rand_radius(), random_color(), cv2.FILLED))
        return parent

    @staticmethod
    def mutate(parent):
        child = Individual(parent.circles)
        for i in range(len(parent.circles)):
            c2 = (child.circles[i][0],
                  rand_radius(), random_color(), cv2.FILLED)
            while np.sum(abs(np.array(c2[2]) - target[c2[0]])) < np.sum(child.circles[i][2] - target[c2[0]]):
                child.circles[i] = (c2[0], c2[1], c2[2], c2[3])
                c2 = (child.circles[i][0],
                      rand_radius(), random_color(), cv2.FILLED)
        return child


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Please provide path to input image')
        sys.exit(0)
    path = sys.argv[1]
    main_image = load_image(path)
    target = average_color(main_image)
    plot_image(main_image)
    res = Population.evolution()

