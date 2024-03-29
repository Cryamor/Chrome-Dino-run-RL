'''
use selenium.webdriver to control dino in Chrome
'''
import os, shutil
import time
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By

class Controller():
    def __init__(self) -> None:
        if not os.path.exists('pic'):
            os.mkdir('pic')
        if os.path.exists('pic/screenshot'):
            shutil.rmtree('pic/screenshot')
        os.mkdir('pic/screenshot')
        if not os.path.exists('pic/plot'):
            os.mkdir('pic/plot')
        
        options = Options()
        options.add_experimental_option('detach', True) # prevent Chrome crash
        options.add_argument("disable-infobars")
        self.driver = webdriver.Chrome(chrome_options = options)
        self.driver.maximize_window()
        try:
            self.driver.get('chrome://dino')
        except WebDriverException:
            pass
        self.driver.execute_script("Runner.config.ACCELERATION=0")
        self.driver.execute_script("document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")
        self.restart()
        self.jump() # start game
      
    def run(self, actions):
        '''control dino movement using
        actions[moveforward, jump, duck]'''
        if actions[0] == 1:
            pass
        elif actions[1] == 1:
            self.jump()
        elif actions[2] == 1:
            self.duck()
            
        # get game score
        score = int(''.join(self.driver.execute_script("return Runner.instance_.distanceMeter.digits;")))    
        # get whether or not dino is dead 
        is_dead = False
        if self.driver.execute_script("return Runner.instance_.crashed;"):
            self.restart()
            is_dead = True
        # get game screenshot image
        image = self.screenshot()
        
        return score, is_dead, image
    
    # encapsulate operations
    def jump(self):
        self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ARROW_UP)
    
    def duck(self):
        self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ARROW_DOWN)
    
    def pause(self):
        return self.driver.execute_script("Runner.instance_.stop();")
    
    def restart(self):
        self.driver.execute_script("Runner.instance_.restart();")
        time.sleep(0.2)
    
    def resume(self):
        return self.driver.execute_script("Runner.instance_.play();")
    
    def stop(self):
        self.driver.close()
    
    def screenshot(self):
        '''use opencv to process game image'''
        area = (0, 0, 150, 450)
        image_b64 = self.driver.execute_script("canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)")
        image = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        
        # save screenshot every 20s
        t_f = time.time()
        if int(t_f) % 20 == 0 and str(t_f).split('.')[-1][0] == '0':
            image2 = Image.open(BytesIO(base64.b64decode(image_b64)))
            image2.save('pic/screenshot/{}.jpg'.format(t_f), format='png') # RGBA cannot be saved as JPEG
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[area[0]: area[2], area[1]: area[3]]
        return image
    