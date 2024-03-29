import torch
import torch.nn as nn
import cv2
import numpy as np
from model import MyModel
import random
import time
from collections import deque
import matplotlib.pyplot as plt

class Agent():
    def __init__(self, mode, fps, checkpoint):
        self.mode = mode
        self.fps = fps
        self.chekpoint =checkpoint
        
        # self.image_size = (84, 84)
        self.image_size = (80, 80)
        self.action_num = 3
        self.image_frame_num = 4  # get frames of images as input
        
        self.batch_size = 32
        self.observe_num = 3200
        self.replay_memory_record = deque()
        self.replay_memory_size = 1e4
        self.gamma = 0.9 # discount factor
        
        # use epsilon-greedy algorithm
        self.start_epsilon = 0.1
        self.end_epsilon = 1e-4
        self.now_epsilon = self.start_epsilon
        self.down_percent = 1e5
        self.save_prob = 0.1 # save probability
        self.replace_iter = 200 # every {} iters, set target_model = model 
        
        self.score = 0
        self.max_score = 0
        self.game_num = 1
        self.iter_num = 0
        
        # draw plot
        self.iter_list = []
        self.loss_list = []
        self.game_list = []
        self.score_list = []
        
        # model
        self.model = MyModel(self.image_size, self.image_frame_num, self.action_num)
        self.target_model = MyModel(self.image_size, self.image_frame_num, self.action_num)
        # use gpu
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.target_model = self.target_model.cuda() if torch.cuda.is_available() else self.target_model   
        self.model.apply(MyModel.initWeights)
        self.target_model.apply(MyModel.initWeights)
        
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_func = nn.MSELoss() 
        # self.loss_func = nn.CrossEntropyLoss()
         
    def preprocess(self, image, size):
        image = cv2.resize(image, size)
        image[image > 0] = 255
        image = np.expand_dims(image, 0)
        return image
    
    def train(self, controller, is_resume = False):
        
        actions = np.array([0] * self.action_num)
        actions[0] = 1
        score, is_dead, image = controller.run(actions)
        image = self.preprocess(image, self.image_size)
        self.input_image = np.tile(image, (self.image_frame_num, 1, 1))
        self.input_image = self.input_image.reshape(1, self.input_image.shape[0], self.input_image.shape[1], self.input_image.shape[2])
        last_time = 0
        
         
        while True:
            # update target_model
            if self.iter_num % self.replace_iter == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            # update iter   
            self.iter_num += 1
            if self.iter_num > self.observe_num:
                self.iter_list.append(self.iter_num)
            
            # randomly or use model to control dino
            actions = np.array([0] * self.action_num)
            if random.random() < self.now_epsilon:
                actions[random.randint(0, self.action_num - 1)] = 1
            else:
                self.model.eval()
                input_image = torch.from_numpy(self.input_image).type(self.FloatTensor)
                with torch.no_grad():
                    preds = self.model(input_image).cpu().data.numpy()
                actions[np.argmax(preds)] = 1
                self.model.train()
                
            if self.game_num > 1500:
                break
                
            # act
            try:
                score, is_dead, image = controller.run(actions)
            except Exception:
                break # when chrome is closed, break
            image = self.preprocess(image, self.image_size)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            input_image_prev = self.input_image.copy()
            # get recent frames of image
            self.input_image = np.append(image, self.input_image[:, :self.image_frame_num-1, :, :], axis=1)
            
            # fps control
            if last_time:
                now_fps = 1 / (time.time() - last_time)
                if now_fps > self.fps:
                    time.sleep(1 / self.fps - 1 / now_fps)
            last_time = time.time()
            
            # reward
            if is_dead:
                reward = -1
                self.game_list.append(self.game_num)
                self.score_list.append(score)
                self.game_num += 1
            else:
                reward = 0.1
                
            # save max score and best model
            self.score = score
            if score > self.max_score:
                self.max_score = self.score
                self.save('checkpoints/best-checkpoint.pth')
            
            # save memory record for training
            if is_dead or random.random() <= self.save_prob:
                self.replay_memory_record.append([input_image_prev, self.input_image, actions, np.array([int(is_dead)]), np.array([reward])])
                print('[Memory Record]:', len(self.replay_memory_record))
            if len(self.replay_memory_record) > self.replay_memory_size:
                self.replay_memory_record.popleft()
                
            # train the model
            loss = torch.Tensor([0]).type(self.FloatTensor)
            if self.iter_num > self.observe_num: # after enough iter, train
                # get minibatch randomly from previous
                minibatch = random.sample(self.replay_memory_record, self.batch_size)
                # unzip
                input_image_prev_, input_image_, actions_, is_dead_, reward_ = zip(*minibatch)
                # comvert to tensor
                input_image_prev_ = torch.from_numpy(np.concatenate(input_image_prev_)).type(self.FloatTensor)
                input_image_ = torch.from_numpy(np.concatenate(input_image_)).type(self.FloatTensor)
                actions_ = torch.from_numpy(np.concatenate(actions_)).type(self.FloatTensor).view(self.batch_size, -1) # 32*3
                is_dead_ = torch.from_numpy(np.concatenate(is_dead_)).type(self.FloatTensor)
                reward_ = torch.from_numpy(np.concatenate(reward_)).type(self.FloatTensor)
                
                with torch.no_grad():
                    targets = reward_ + self.gamma * self.target_model(input_image_).max(-1)[0] * (1 - is_dead_)
                    targets = targets.detach()
                preds = torch.sum(self.model(input_image_prev_) * actions_, dim = 1) # 32*1
                
                loss = self.loss_func(preds, targets)
                self.loss_list.append(loss.cpu().detach().numpy())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

            # update epsilon
            if self.now_epsilon > self.end_epsilon and self.iter_num > self.observe_num:
                self.now_epsilon -= (self.start_epsilon - self.end_epsilon) / self.down_percent
            
            # save the model
            if self.iter_num % 5000 == 0:
                self.save(self.chekpoint)
                
            # print message
            print('[State]: train, [Games]: {0}, [Iter]: {1}, [Score]: {2}, [MaxScore]: {3}, [Action]: {4}, [Epsilon]: {5}, [Loss]: {6}'.\
                format(self.game_num, self.iter_num, self.score, self.max_score, np.argmax(actions), self.now_epsilon, loss.item()))   
            
        plt.figure(figsize = (14, 10), dpi = 100)
        plt.subplot(1,2,1)
        try:
            score_line.remove(score_line[0])
        except Exception:
            pass
        self.iter_list.pop() # delete the last iter with no loss 
        score_line = plt.plot(self.game_list, self.score_list)
        plt.title('Score')
        plt.xlabel('game')
        plt.ylabel('score')
        plt.xlim(1, len(self.game_list))
        plt.ylim(0, max(self.score_list) + 500)
        
        plt.subplot(1,2,2)
        try:
            loss_line.remove(loss_line[0])
        except Exception:
            pass
        loss_line = plt.plot(self.iter_list, self.loss_list, 'r')
        plt.title('Loss')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.xlim(self.observe_num, len(self.iter_list))
        plt.ylim(0, max(self.loss_list))
        plt.savefig('pic/plot/train-{}.png'.format(time.time()))
        plt.show()
                
    def test(self, controller):
        actions = np.array([0] * self.action_num)
        actions[0] = 1
        score, is_dead, image = controller.run(actions)
        image = self.preprocess(image, self.image_size)
        self.input_image = np.tile(image, (self.image_frame_num, 1, 1))
        self.input_image = self.input_image.reshape(1, self.input_image.shape[0], self.input_image.shape[1], self.input_image.shape[2])
        last_time = 0
        
        while True:
            actions = np.array([0] * self.action_num)
            self.model.eval()
            input_image = torch.from_numpy(self.input_image).type(self.FloatTensor)
            with torch.no_grad():
                preds = self.model(input_image).cpu().data.numpy()
            actions[np.argmax(preds)] = 1
            
            # act
            try:
                score, is_dead, image = controller.run(actions)
            except Exception:
                break
            image = self.preprocess(image, self.image_size)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            self.input_image = np.append(image, self.input_image[:, :self.image_frame_num-1, :, :], axis=1)
			
            if is_dead:
                self.game_list.append(self.game_num)
                self.score_list.append(score)
                self.game_num += 1
            if last_time:
                now_fps = 1 / (time.time() - last_time)
                if now_fps > self.fps:
                    time.sleep(1 / self.fps - 1 / now_fps)
            last_time = time.time()
            
            self.score = score
            if score > self.max_score:
                self.max_score = score
                
            # print message
            print('[State]: test, [Games]: {0}, [Score]: {1}, [MaxScore]: {2}, [Action]: {3}'.\
                format(self.game_num, self.score, self.max_score, np.argmax(actions)))   
            
            
        # draw plot
        plt.figure(figsize = (6, 6), dpi = 96)
        try:
            score_line.remove(score_line[0])
        except Exception:
            pass
        score_line = plt.plot(self.game_list, self.score_list)
        plt.title('Score')
        plt.xlabel('game')
        plt.ylabel('score')
        plt.xlim(1, len(self.game_list))
        plt.ylim(0,max(self.score_list) + 500)
        plt.savefig('pic/plot/test-{}.png'.format(time.time()))
        plt.show()
                
    
    def save(self, checkpoint):
        print('Saving checkpoints into {}...'.format(checkpoint))
        torch.save(self.model.state_dict(), checkpoint)
	
    def load(self, checkpoint):
        print('Loading checkpoint from {}...'.format(checkpoint))
        self.model.load_state_dict(torch.load(checkpoint))