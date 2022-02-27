from tkinter import E

import pygame
import sys
import random
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.models import InputLayer
from tensorflow.keras import initializers
import numpy as np
import math

#set up window
SCREEN_WIDTH = 250
SCREEN_HEIGHT = 250

checkPoints = []
NUM_CHECKPOINTS = 3
players = []
NUM_PLAYERS = 125
endPoints = []
NUM_ENDPOINTS = 1

DIM = 35

TICK_MAX = 4000
START_TICKS = 0

class Player():
    def __init__(self, x, y, width, height, brain):
        self.colour = (0, 250, 0)
        self.width, self.height = width, height
        self.collected = [False]*NUM_CHECKPOINTS
        self.collectedCount = 0
        self.score = 0
        self.fitness = 0
        self.x = x
        self.y = y
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.vel = 20

        if brain is not None:
            self.brain = brain
        else:
            self.brain = self.makeBrain(NUM_CHECKPOINTS)
        
    # def move(self):
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_w]:
    #         if (self.y > 0):
    #             self.y -= self.vel
    #         else:
    #             self.y = 0
    #     if keys[pygame.K_s]:
    #         if (self.y >= SCREEN_HEIGHT-self.height):
    #             self.y = SCREEN_HEIGHT-self.height
    #         else:
    #             self.y += self.vel
    #     if keys[pygame.K_a]:
    #         if (self.x > 0):
    #             self.x -= self.vel
    #         else:
    #             self.x = 0
    #     if keys[pygame.K_d]:
    #         if (self.x >= SCREEN_WIDTH-self.width):
    #             self.x = SCREEN_WIDTH-self.width
    #         else:
    #             self.x += self.vel

    #     self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
    #     self.checkCollisions()

    def move(self):

        move = self.think()

        # Move up
        if move == 0:
            if (self.y > 0):
                self.y -= self.vel
            else:
                self.y = 0
        # Move down
        if move == 1:
            if (self.y >= SCREEN_HEIGHT-self.height):
                self.y = SCREEN_HEIGHT-self.height
            else:
                self.y += self.vel
        # Move left
        if move == 2:
            if (self.x > 0):
                self.x -= self.vel
            else:
                self.x = 0
        # Move right
        if move == 3:
            if (self.x >= SCREEN_WIDTH-self.width):
                self.x = SCREEN_WIDTH-self.width
            else:
                self.x += self.vel

        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.checkCollisions()

    def checkCollisions(self):

        for i, cp in enumerate(checkPoints):
            if self.rect.colliderect(cp.rect):
                if not self.collected[i]:
                    self.collected[i] = True
                    
                    #print("START_TICKS = " +str(START_TICKS))
                    self.score += abs(pygame.time.get_ticks()-START_TICKS - TICK_MAX-START_TICKS)%TICK_MAX
                    #print(self.score)
                    self.collectedCount+=1
                    #print(self.collectedCount)
        
        for ep in endPoints:
            if self.collectedCount == NUM_CHECKPOINTS and self.rect.colliderect(ep.rect):
                self.score+=TICK_MAX/2 * pow(NUM_CHECKPOINTS,2)

    def draw(self, surface):
        pygame.draw.rect(surface, self.colour, self.rect)

    def makeBrain(self, checkpointNum):

        #define brain as sequential nueral net
        brain=Sequential()

        #hidden layers
        brain.add(Dense(6, input_dim = NUM_CHECKPOINTS, activation='relu',kernel_initializer=initializers.random_normal(), bias_initializer=initializers.zeros())) 
        brain.add(Dense(6, activation='relu', kernel_initializer=initializers.random_normal(), bias_initializer=initializers.zeros())) 

        #output layer - 4 nodes for 4 different possible moves
        brain.add(Dense(4, activation='softmax'))
    
        return brain

    def think(self): #endpoint and boxes are global variables 

        #declare inputs 
        inputs=[]

        # append current location of box and end point
        #inputs.append(self.x)
        #inputs.append(self.y)


        for i, cp in enumerate(checkPoints):
            inputs.append(calc_euclidean((self.x, self.y), (cp.x, cp.y)))
            #inputs.append((self.collected[i]))

        inputs = np.array(inputs).reshape(1, -1)

        #Get the output from the net
        # outputs = self.brain.predict(inputs)

        outputs = self.brain(inputs, training=False)
        #find output with what the brain thinks the best move is
        prediction = np.argmax(outputs)

        #make the move (move is integer between 0 and 3)
        return prediction
    
class EndPoint():
    def __init__(self):
        self.colour = (255,0,0)
        self.width = self.height = DIM
        self.validSpawn()

    def validSpawn(self):
        self.x = random.randint(0,SCREEN_WIDTH-self.width)
        self.y = random.randint(0,SCREEN_HEIGHT-self.height)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        if self.rect.colliderect(players[0].rect):
            self.validSpawn()

    def draw(self, surface):
        pygame.draw.rect(surface, self.colour, self.rect)

class CheckPoint():
    def __init__(self):
        self.colour = (0,0,255) #put a border on each checkpoint for visual
        self.width = self.height = DIM
        self.validSpawn()
        #write something to handle not spawning on top of other checkpoints/player/end

    def validSpawn(self):
        self.x = random.randint(0,SCREEN_WIDTH-self.width)
        self.y = random.randint(0,SCREEN_HEIGHT-self.height)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

        if self.rect.colliderect(endPoints[0].rect) or self.rect.colliderect(players[0].rect):
            self.validSpawn()

        for cp in checkPoints:
            if self.rect.colliderect(cp.rect):
                self.validSpawn()

    def draw(self, surface):
        pygame.draw.rect(surface, self.colour, self.rect)

def findCandidates():

    scoreSum = 0

    for player in players:
        scoreSum+=player.score

    if scoreSum == 0:
        return [players[0]]

    candidates = []

    sum = 0

    # Normalises the player fitness.
    for player in players:
        player.fitness = player.score / scoreSum
        if player.fitness != 0:
            sum+=player.fitness
            #print(player.fitness)
            candidates.append(player)

    #print("TOTAL: " + str(sum))

    return candidates
    
def selectCandidate(candidates):

    rand = random.random()
    cand = candidates[0]
    sum = 0

    for player in candidates:
        sum+=player.fitness
        if rand <= sum:
            cand=player

    #print("SUM: " + str(sum))

    if cand:
        return cand
    else:   
        print("SOMETHING WENT WRONG")

def mutate(brain, rate):

    # first iterate through the layers
    for j, layer in enumerate(brain.layers):
        new_weights_for_layer = []
        # each layer has 2 matrices, one for connection weights and one for biases
        # then iterate though each matrix
        for weight_array in layer.get_weights():
            
            # save their shape
            save_shape = weight_array.shape
            # reshape them to one dimension
            one_dim_weight = weight_array.reshape(-1)

            for i, weight in enumerate(one_dim_weight):
                # mutate them like i want
                if random.uniform(0, 1) <= rate:
                    # maybe dont use a complete new weigh, but rather just change it a bit
                    one_dim_weight[i] += random.uniform(0, 2) - 1

            # reshape them back to the original form
            new_weight_array = one_dim_weight.reshape(save_shape)
            # save them to the weight list for the layer
            new_weights_for_layer.append(new_weight_array)

        # set the new weight list for each layer
        brain.layers[j].set_weights(new_weights_for_layer)

    return brain

def evolve():

    global players

    # Makes player score exponential. Might help might not.
    for player in players:
        player.score = pow(player.score, 2)

    candidates = findCandidates()

    cand = selectCandidate(candidates)
    
    return cand.brain

def generatePlayers(brain):

    global players
    players = []

    playerWidth = 15
    playerHeight = 15
    playerX = random.randint(0,SCREEN_WIDTH-playerWidth)
    playerY = random.randint(0,SCREEN_HEIGHT-playerHeight)

    for i in range(NUM_PLAYERS):
        # If there is a brain passed as an input, use it to generate the players. Otherwise the players generate brains themselves.
        if brain is not None:
            players.append(Player(playerX, playerY, playerWidth, playerHeight, mutate(brain, .9)))
        else:
            players.append(Player(playerX, playerY, playerWidth, playerHeight, brain))

def calc_euclidean(point_1, point_2):
    """
    Calculates the euclidean distance between two points
    """

    return math.sqrt(pow(point_1[0] - point_2[0], 2) + pow(point_1[1] - point_2[1], 2))        


def main():

    pygame.init()

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    surface = pygame.Surface(screen.get_size())

    brain = None


    while True:

        global START_TICKS
        START_TICKS=pygame.time.get_ticks()

        generatePlayers(brain)
        #print(len(players))

        global endPoints
        endPoints = []

        for i in range(NUM_ENDPOINTS):
            endPoints.append(EndPoint())

        global checkPoints
        checkPoints = []

        for i in range(NUM_CHECKPOINTS):
            checkPoints.append(CheckPoint())

        while (True):

            ticks = pygame.time.get_ticks()-START_TICKS

            if ticks > TICK_MAX:
                brain = evolve()
                #print(brain)
                break

            #print("TICK: " + str(ticks))

            clock.tick(60)

            surface.fill((200, 200, 200))

            for p in players:
                p.move()
                p.draw(surface)

            for cp in checkPoints:
                cp.draw(surface)

            for ep in endPoints:
                ep.draw(surface)
            
            screen.blit(surface, (0, 0))
            pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

main()