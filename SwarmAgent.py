import sys
import os
import numpy as np
import cv2
import copy
import math
from Core import Object

from EnforcementLearning import *

from utils.Vector import Vector3 as Vector

PARTICLE_UPPER_BOUNDS = Vector( 1,  1,  1)
PARTICLE_LOWER_BOUNDS = Vector(-1, -1, -1)

PARTICLE_UPPER_BOUNDS_POSITION = PARTICLE_UPPER_BOUNDS.clone()
PARTICLE_LOWER_BOUNDS_POSITION = PARTICLE_UPPER_BOUNDS.clone()

PARTICLE_UPPER_BOUNDS_SPEED = PARTICLE_UPPER_BOUNDS.clone()
PARTICLE_LOWER_BOUNDS_SPEED = PARTICLE_LOWER_BOUNDS.clone()

INTERIA_FACTOR = 0.98
COGNITIVE_WEIGHT = 0.1
SOCIAL_WEIGHT = 0.001
GLOBAL_WEIGHT = 0

RANGE_OF_DISTANCE_BETWEEN_PARTICLES = 20

class SwarmAgent(Agent):
    def __init__(self, numParticles=100, mesh=None):
        super(SwarmAgent, self).__init__("SwarmAgent")
        self.swarm = Swarm(
            numParticles=numParticles,
            fitnessFunction=ImageFitnessFunction
        )
        for particle in self.swarm.particles:
            particle.object = Object(meshes=[mesh])

    @property
    def objects(self):
        return [particle.object for particle in self.swarm.particles]

    def update(self):
        self.swarm.update()
        for particle in self.swarm.particles:
            particle.object.setPosition(particle.position.toList())

class FitnessFunction:
    def __init__(self):
        pass

    @staticmethod
    def getFitness(particle):
        return 0

class Particle:
    def __init__(self,
        initialPosition=Vector(),
        initialScale=0.0,
        initialSpeed=Vector(),
        fitnessFunction=FitnessFunction):
        self.position = initialPosition
        self.scale = initialScale
        self.speed = initialSpeed
        self.fitness = 0.0
        self.fitnessFunction = fitnessFunction
        self.bestPosition = Vector()
        self.bestFitness = -sys.float_info.max
        self.orderIndex = Vector()

    def __repr__(self):
        return "position : {!r}, scale : {!r}, speed : {!r}".format(self.position, self.scale, self.speed)

    def __str__(self):
        return "position : {!r}, scale : {!r}, speed : {!r}".format(self.position, self.scale, self.speed)

    def setSpeed(self, speed):
        self.speed = speed

    def setFitness(self, fitness):
        self.fitness = fitness
        if  fitness > self.bestFitness:
            self.bestFitness = fitness
            self.bestPosition = self.position.clone()

    def setFitnessFunction(self, fitnessFunction):
        self.fitnessFunction = fitnessFunction

    def calcFitness(self):
        fitness = self.fitnessFunction.getFitness(self)
        self.setFitness(fitness)
        self.position = self.position + self.speed

    def updateParticleSpeed(self):
        self.setSpeed(
            INTERIA_FACTOR * self.speed
            + COGNITIVE_WEIGHT * (self.bestPosition - self.position)
        )

    def update(self):
        self.calcFitness()
        self.updateParticleSpeed()

class Swarm:
    def __init__(self,
        numParticles=10,
        fitnessFunction=FitnessFunction):
        self.particles = [
            Particle(
                initialPosition = Vector.randfloat(
                    PARTICLE_LOWER_BOUNDS_POSITION,
                    PARTICLE_UPPER_BOUNDS_POSITION
                ),
                initialSpeed = Vector.randfloat(
                    PARTICLE_LOWER_BOUNDS_SPEED,
                    PARTICLE_UPPER_BOUNDS_SPEED
                ),
                fitnessFunction = fitnessFunction
            ) for x in range(numParticles)
        ]
        self.bestPosition = Vector()
        self.bestFitness = -sys.float_info.max
        self.order_x = None
        self.order_y = None
        self.calcParticleOrder()

    def setFitnessFunction(self, fitnessFunction):
        for particle in self.particles:
            particle.setFitnessFunction(fitnessFunction)

    def calcParticleOrder(self):
        self.order_x = self.calcParticleOrder_by(orderedBy=0)
        self.order_y = self.calcParticleOrder_by(orderedBy=1)

    def calcParticleOrder_by(self, orderedBy=0):
        order = sorted(self.particles, key=lambda u: u.position[orderedBy])
        for i in range(len(order)):
            order[i].orderIndex[orderedBy] = i
        return order

    def calcFitness(self):
        for particle in self.particles:
            particle.calcFitness()
            if particle.bestFitness > self.bestFitness:
                self.bestFitness = particle.bestFitness
                self.bestPosition = particle.bestPosition

    def updateParticleSpeed(self):
        for particle in self.particles:
            particle.setSpeed(
                INTERIA_FACTOR * particle.speed
                + COGNITIVE_WEIGHT * (particle.bestPosition - particle.position) * sigmoid(particle.bestFitness - self.bestFitness)
                + SOCIAL_WEIGHT * (self.bestPosition - particle.position) * sigmoid(self.bestFitness - particle.bestFitness)
            )

    def update(self):
        self.calcFitness()
        self.updateParticleSpeed()
        self.calcParticleOrder()
        #print(self.bestPosition)

def sigmoid(x, a=5):
    return 1 / (1 + math.exp(-a * x))

class Multiswarm:
    def __init__(self,
        numSwarms=10,
        particlesPerSwarm=10,
        fitnessFunction=FitnessFunction):
        self.swarms = [
            Swarm(
                numParticles = particlesPerSwarm,
                fitnessFunction = fitnessFunction
            ) for x in range(numSwarms)
        ]
        self.bestPosition = Vector()
        self.bestFitness = -sys.float_info.max

    def setFitnessFunction(self, fitnessFunction):
        for swarm in self.swarms:
            swarm.setFitnessFunction(fitnessFunction)

    def calcFitness(self):
        for swarm in self.swarms:
            swarm.calcFitness()
            if swarm.bestFitness > self.bestFitness:
                self.bestFitness = swarm.bestFitness
                self.bestPosition = swarm.bestPosition

    def updateParticleSpeed(self):
        for swarm in self.swarms:
            for particle in swarm.particles:
                particle.setSpeed(
                    INTERIA_FACTOR * particle.speed
                    + COGNITIVE_WEIGHT * (particle.bestPosition - particle.position)
                    + SOCIAL_WEIGHT * (swarm.bestPosition - particle.position)
                    + GLOBAL_WEIGHT * (self.bestPosition - particle.position)
                )

    def update(self):
        self.calcFitness()
        self.updateParticleSpeed()

class ImageFitnessFunction(FitnessFunction):
    image = None
    swarm = None
    @classmethod
    def setImage(cls, image):
        cls.image = copy.deepcopy(image)

    @classmethod
    def setSwarm(cls, swarm):
        cls.swarm = swarm

    @staticmethod
    def getFitness(particle):
        image = ImageFitnessFunction.image
        swarm = ImageFitnessFunction.swarm

        return Vector.randfloat(Vector(1,0,0)).x

        #フィットネス値
        fitness = 0

        #位置による適応度
        fit_pos = 0
        x = int(particle.position.x)
        y = int(particle.position.y)
        img_h = len(image)
        img_w = len(image[0])
        if x < 0 or y < 0 or x >= img_w or y >= img_h:
            fit_pos = 0
        elif image[y, x, 3] > 0.5:
            fit_pos = 1
        else:
            fit_pos = 0

        #周囲の粒子の位置による適応度
        count = 0
        index = particle.orderIndex[0]
        pos = particle.position
        range = RANGE_OF_DISTANCE_BETWEEN_PARTICLES
        while swarm.order_x[index].position.x < (pos.x + range):
            index += 1
            if index >= len(swarm.order_x):
                break
            t_pos = swarm.order_x[index].position
            if (pos - t_pos).length() < range:
                count += 1 / math.exp((pos - t_pos).length())

        index = particle.orderIndex[0]
        while swarm.order_x[index].position.x > (pos.x - range):
            index -= 1
            if index < 0:
                break
            t_pos = swarm.order_x[index].position
            if (pos - t_pos).length() < range:
                count += 1 / math.exp((pos - t_pos).length())

        try:
            fit_other = 1 - 1 / (1 + math.exp(-5 * (count / 100 - 1)))
            #fit_other = 1 - math.exp(5 * count / 100)
        except Exception:
            fit_other = 0

        return 1.0 * fit_pos + 0.5 * fit_other

'''
def test2():
    img = cv2.imread('input5.png', cv2.IMREAD_UNCHANGED)
    ImageFitnessFunction.setImage(img)
    #cv2.imshow("test", img)

    swarm = Swarm(
        numParticles=500,
        fitnessFunction=ImageFitnessFunction
    )

    ImageFitnessFunction.setSwarm(swarm)

    while True:
        swarm.update()
        render = np.zeros((len(img), len(img[0]), 3), np.uint8)
        render += img[:,:,:3]
        swarm.bestFitness *= 0.1
        for particle in swarm.particles:
            #particle.bestFitness *= 1.0
            drawCircle(render, particle.position.x, particle.position.y, thickness=2)

        cv2.imshow("test", render)
        key = cv2.waitKey(1)
        print(
            'BEST - fitness:{:>12.3f}, position:({:>12.3f},{:>12.3f}), p_pos=({},{})'.format(
                swarm.bestFitness,
                swarm.bestPosition.x,
                swarm.bestPosition.y,
                swarm.particles[0].position.x,
                swarm.particles[0].position.y
            )
        )
        if key == ord('q'):
            break

    print(
        'Final Score - fitness:{:>12.3f}, position:({:>12.3f},{:>12.3f})'.format(
            swarm.bestFitness,
            swarm.bestPosition.x,
            swarm.bestPosition.y
        )
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawCircle(img, x, y, r=1, color=(0, 0, 255), thickness=-1):
    x = int(x)
    y = int(y)
    if x < 0 or y < 0 or x >= len(img[0]) or y >= len(img):
        return
    cv2.circle(img, (x, y), r, color, thickness=thickness)

if __name__ == "__main__":
    test2()
'''
