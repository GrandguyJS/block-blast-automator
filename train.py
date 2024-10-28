from main import BlockBlast
from ai import NeuralNetwork

import multiprocessing as mp
import numpy as np
import random
import os
import copy
import time

def shape_to_binary(integer):
    binary_str = bin(integer)[2:]
    binary_list = [int(bit) for bit in binary_str]
    binary_list.extend([0 for _ in range(25-len(binary_list))])
    return binary_list

def create_input(game, shapes):
    _output = np.array(game.grid).flatten()
    _shapes = np.array([shape_to_binary(num) for num in shapes]).flatten()

    output = np.concatenate((_output, _shapes))

    return output

def decode_data(output_bits):
    output = [np.argmax(output_bits[:6])-1]

    coord_bits = output_bits[6:]

    for i in range(6):
        start = i * 8
        end = start + 8
        coord = np.argmax(coord_bits[start:end])-1
        output.append(coord)

    return output

path = "model/parameters.pkl"

num = 200
ai = NeuralNetwork([139, 128 , 64  , 32 , 21], num)
ai.load(path)
game = BlockBlast()

def play():
    games = []
    for i in range(0, num):
        games.append([copy.deepcopy(game), 1, 0])

    while sum([x[1] for x in games]) != 0:
        shapes = random.sample(game.shapes, 3)
        shapes_order = [[shapes[0], shapes[1], shapes[2]], [shapes[0], shapes[2], shapes[1]], [shapes[1], shapes[0], shapes[2]],
                        [shapes[1], shapes[2], shapes[0]], [shapes[2], shapes[0], shapes[1]], [shapes[2], shapes[1], shapes[0]]
                        ]
        input_ai = create_input(game, shapes)

        _output = ai.forward(input_ai)
        for i,output in enumerate(_output):
            output = output.flatten().tolist()
            data = decode_data(output)

            action = data[0]
            coords = data[1:]

            shapes = shapes_order[action]
            for j in range(0, len(shapes)):
                if games[i][1] == 1:
                    move = games[i][0].put(shapes[j], coords[2*j], coords[2*j+1])
                    if not move:
                        games[i][1] = 0
                    else:
                        games[i][2] += 1
    return games

def train(epochs = 1):
    for i in range(0, epochs*1000):
        games = play()
        index = np.argmax([x[2] for x in games])
        ai.selection(index)
        print(f"Epoch {i+1}: {games[index][2]} moves")
    ai.save(path)

try:
    train()
finally:
    ai.save(path)