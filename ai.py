from main import BlockBlast

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Define the neural network architecture
class BlockBlastNet(nn.Module):
    def __init__(self):
        super(BlockBlastNet, self).__init__()
        
        self.fc1 = nn.Linear(139, 512)  
        self.ln1 = nn.LayerNorm(512)  # Use LayerNorm instead of BatchNorm
        
        self.fc2 = nn.Linear(512, 1024)  
        self.ln2 = nn.LayerNorm(1024)  # Use LayerNorm instead of BatchNorm
        
        self.fc3 = nn.Linear(1024, 512)  
        self.ln3 = nn.LayerNorm(512)  # Use LayerNorm instead of BatchNorm
        
        self.fc4 = nn.Linear(512, 24)  
        self.dropout = nn.Dropout(p=0.5)  
        
        self.relu = nn.ReLU()
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))  # Apply first layer and layer norm
        x = self.dropout(x)                    # Apply dropout
        x = self.relu(self.ln2(self.fc2(x)))  # Apply second layer and layer norm
        x = self.dropout(x)                    # Apply dropout
        x = self.relu(self.ln3(self.fc3(x)))  # Apply third layer and layer norm
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return np.array(x)[0]

    def get_input(self, grid, shape1, shape2, shape3):
        _input = np.array(grid).flatten()

        shapes = [shape1, shape2, shape3]
        for shape in shapes:
            bytes_shape = [int(bit) for bit in bin(shape)[2:]]
            bytes_shape.extend([0 for _ in range(25-len(bytes_shape))])
            _input = np.append(_input, np.array(bytes_shape))

        return _input

    def train(self, generations = 10000):
        for i in range(generations):
            self.game = BlockBlast()
            shapes = random.sample(self.game.shapes, 3)
            _input = self.get_input(self.game.grid, shapes[0], shapes[1], shapes[2])

            input_tensor = torch.tensor(_input, dtype=torch.float32).unsqueeze(0)  # Add a batch dimension

            with torch.no_grad(): 
                output = self.forward(input_tensor)

            action_vector = np.round(output.squeeze()).astype(int).tolist()

            actions = []
            actions.extend(action_vector[:8])
            actions.extend(action_vector[8:16])
            actions.extend(action_vector[16:])

            num1, num2, num3 = int(''.join(map(str, actions[:2])), 2), int(''.join(map(str, actions[8:10])), 2), int(''.join(map(str, actions[16:18])), 2)
            x1, x2, x3 = int(''.join(map(str, actions[2:5])), 2), int(''.join(map(str, actions[10:13])), 2), int(''.join(map(str, actions[18:21])), 2)
            y1, y2, y3 = int(''.join(map(str, actions[5:8])), 2), int(''.join(map(str, actions[13:16])), 2), int(''.join(map(str, actions[21:24])), 2)

            if num1 > 2 or num2 > 2 or num3 > 2 or num1 == num2 or num1 == num3 or num2 == num3 or num1 == num3:
                loss = torch.tensor(10.0, dtype=torch.float32, requires_grad=True)
            else:
                loss = 0.0

                if not self.game.put(shapes[num1], x1, y1):
                    loss += 1.0
                if not self.game.put(shapes[num2], x2, y2):
                    loss += 1.0
                if not self.game.put(shapes[num3], x3, y3):
                    loss += 1.0


                loss = torch.tensor(loss / (3.0 + 1e-8), dtype=torch.float32, requires_grad=True)

            # Backpropagation
            self.optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Compute gradients
            self.optimizer.step()  # Update weights

            print(f"Generation {i + 1}/{generations}, Loss: {loss.item()}, Score: {self.game.score}") 
            print(self.game) 

loaded_model = BlockBlastNet()
loaded_model.load_state_dict(torch.load('block_blast_model.pth'))
loaded_model.train()

torch.save(loaded_model.state_dict(), 'block_blast_model.pth')