# File name : snake.py
# Author : Joseph Kang
# Last Updated : 2021-09-17

import pygame
import random
import numpy as np
from numpy import array as Vec
from model import DQN
import torch
import copy
import matplotlib.pyplot as plt
import os

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

BoardX = 30 #게임 사이즈
BoardY = 30

PixelPerBlock = 30

screen = pygame.display.set_mode(
    (BoardX*PixelPerBlock, BoardY*PixelPerBlock), pygame.DOUBLEBUF)

pygame.display.set_caption("Snake Game in RL")

clock = pygame.time.Clock()

framerate = 100

delta = [ #방향당 위치 변화값
    Vec([1, 0]),  # Right
    Vec([0, 1]),  # Down
    Vec([-1, 0]),  # Left
    Vec([0, -1])  # Up
]

KEY2DIR = { 
    pygame.K_UP: 3,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 0
}


def Opposite(dir): #정면방향 기준 뒤쪽 방향 (절대방향) 리턴
    return (dir+2) % 4

def getDir(dir): #정면방향 기준 [좌측, 정면, 우측] (절대방향) 리턴
    foward = dir
    right = (dir+1) % 4
    left = (dir-1) % 4
    return [left, foward, right]

def norm(vec):
    return abs(vec[0])+abs(vec[1])

def DrawBlock(position, color):
    block = pygame.Rect(position[0]*PixelPerBlock, position[1]
                        * PixelPerBlock, PixelPerBlock, PixelPerBlock)
    pygame.draw.rect(screen, color, block)

def isEqual(vec1, vec2): #두 벡터가 방향이 같은지 확인 
    vec2=vec2/norm(vec2)
    if np.array_equal(vec1,vec2):
        return True
    return False

score_history = []

class Snake:
    def __init__(self): #게임 초기상태
        self.board=np.zeros((BoardX,BoardY))
        self.body = [Vec([BoardX//2, BoardY//2]), Vec([BoardX//2, BoardY//2])]
        self.board[BoardX//2][BoardY//2]=2
        self.dir = 0
        self.last_dir = 0
        self.consume = False
        self.genApple()
        self.score = 0
        self.last_distance=self.apple-self.body[0]

    def Draw(self): #화면 출력
        for position in self.body:
            DrawBlock(position, WHITE)
        DrawBlock(self.apple, RED)

    def MoveSnake(self): #현재 dir로 이동
        head = self.body[0]+delta[self.dir]
        self.body.insert(0, head)
        if self.isOutOfBoard(head):
            return
        self.board[head[0],head[1]]+=1
        if np.array_equal(head, self.apple):
            self.consume = True
            self.score += 1
            self.genApple()
        else:
            tail=self.body.pop()
            self.board[tail[0]][tail[1]]-=1

    def isDead(self): #죽었는지 확인
        head = self.body[0]
        if self.isOutOfBoard(head):
            return True
        if self.board[head[0]][head[1]]>1:
            return True
        return False

    def changeDir(self, dir_NEW): #절대방향으로 방향 변경
        self.dir = dir_NEW

    def genApple(self): #사과 생성
        while True:
            temp = random.randrange(0, BoardX*BoardY)
            apple = Vec([temp % BoardX, temp//BoardX])
            if apple[0]==0 or apple[0]==BoardX-1 or apple[1]==0 or apple[1]==BoardY-1:
                continue
            if self.board[apple[0]][apple[1]]>0:
                continue
            self.apple = apple
            self.last_distance=self.apple-self.body[0]
            return

    def getState(self): #현재 state 리턴  
        head = self.body[0]
        dir = getDir(self.dir)  # left,foward,right
        pos=copy.deepcopy(head)+delta[dir[0]]*4+delta[dir[1]]*4
        grid=np.zeros((9,9))
        for i in range(9):
            for j in range(9):
                if self.isOutOfBoard(pos):
                    grid[i][j]=0
                elif self.board[pos[0],pos[1]]>0:
                    grid[i][j]=0
                else: 
                    grid[i][j]=1
                pos+=delta[dir[2]]
            pos+=delta[dir[0]]*9-delta[dir[1]]
        appledir = [0, 0, 0, 0]
        toapple = self.apple-head
        for i in range(3):
            if np.inner(toapple, delta[dir[i]]) > 0:
                appledir[i] = 1
        appledir[1]=isEqual(delta[dir[1]],toapple)
        appledir[3]=isEqual(delta[Opposite(dir[1])],toapple)
        return torch.FloatTensor(list(np.ravel(grid))+appledir)

    def getReward(self): #사과 먹었으면 50점 죽으면 -100점 가까워지면 5점 멀어지면 -2점
        if self.consume:
            self.consume = False
            return 50 * len(self.body)
        if self.isDead():
            return -1000

        now_distance = self.apple - self.body[0]
        if norm(now_distance) < norm(self.last_distance):
            self.last_distance = now_distance
            return 5
        else:
            return -1
        self.last_distance = now_distance

        return -0.777

    def isOutOfBoard(self, position): #해당 좌표가 보드 밖으로 나갔는지 확인
        if position[0] < 0 or position[0] >= BoardX or position[1] < 0 or position[1] >= BoardY:
            return True
        return False

def train(episode):
    global framerate
    Game = Snake() 
    agent = DQN(episode)

    for i in range(episode):
        while True:
            clock.tick(framerate)  #딜레이
            screen.fill(BLACK)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_SPACE:
                        framerate=110-framerate

            dirs = getDir(Game.dir)
            arr = [Game.getState()]
            action = agent.select_action(arr[0])

            Game.changeDir(dirs[action])
            Game.MoveSnake()
            arr.append(Game.getState())
            reward = Game.getReward()

            agent.memorize(arr[0], action, reward, arr[1])
            agent.optimize_model()
            

            if Game.isDead():
                score_history.append(Game.score)
                agent.decay_epsilon()
                Game.__init__()
                break

            Game.Draw()
            pygame.display.flip()
        print(f"{i+1}/{episode} - 점수 : {score_history[i]} / 최고 점수 : {max(score_history)}")

    plt.plot(score_history)
    plt.title(f"Result of Snake Game in RL that {episode} times learning")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.show()

if __name__ == "__main__":
    os.system("clear")
    print("""


Title:\tSnake Game in RL (DQN)
Author:\tJoseph Kang, Ted Song
Code:\thttps://github.com/kitae0522/AI6-DKSH-Hackathon
    
    
    """)
    train(int(1e3))