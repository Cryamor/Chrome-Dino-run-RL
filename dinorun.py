import os
import argparse
from agent import Agent
from controller import Controller

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', help='Choose MODE <train> or <test>', default='train', type=str)
    parser.add_argument('--resume', dest='resume', help='load the training history and continue training', action='store_true')
    args = parser.parse_args()
    mode = args.mode.lower()
    assert mode in ['train', 'test'], '--mode should be <train> or <test>'
 
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    checkpoint = 'checkpoints/checkpoint.pth'
    best_checkpoint = 'checkpoints/best-checkpoint.pth'
    agent = Agent(mode, 30, checkpoint)
    
    # when test or resume train, load checkpoint file
    if os.path.isfile(checkpoint):
        if mode == 'test' or (args.resume and mode == 'train'):
            agent.load(best_checkpoint)
            
    controller = Controller()
    if mode == 'train':
        if args.resume:
            agent.train(controller, True)
        else:
            agent.train(controller)
    else:
        agent.test(controller)
        
    