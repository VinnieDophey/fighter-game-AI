"""
Summative - AI in PyGame
ICS3U-03
Vincent Pham
figher game AI file (game file)
History: Version 1 - May 30, 2025
Version 2 - June 3rd - removed a print that was used for debugging 
"""

# BASICS - Import 
import pygame
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from collections import deque  

pygame.init()

# Game parameters (constants)
FPS = 120
SCREENX = 1920
SCREENY = 1080

# Allow the user to downgrade resolution to meet with ICS3U - Final Project 24-25 requirements.
resChoice = input("\nThis program was built and trained using a 1920 x 1080 resolution. \nHowever, to meet requirements, you may chooose to downscale it to 1000 x 850. \nWARNING: Downscaling may cause the models to not perform as well as on a 1920 x 1080 resolution.\nEnter Y to downscale to 1000 x 850, N to keep original resolution: ").lower()
while resChoice != "y" and resChoice != 'n':
    print("Invalid input! Please answer (Y/N)")
    resChoice = input("\nThis program was built and trained using a 1920 x 1080 resolution. \nHowever, to meet requirements, you may chooose to downscale it to 1000 x 850. \nWARNING: Downscaling may cause the models to not perform as well as on a 1920 x 1080 resolution.\nEnter Y to downscale to 1000 x 850, N to keep original resolution: ").lower()
if resChoice == 'y':
    SCREENX = 1000
    SCREENY = 850
normalFont = pygame.font.SysFont('Raleway', 50)

# SPRITE CLASSES - One class for every sprite (or group of same sprites)

# Player (Played by AI) Sprite:
class MovingSprite(pygame.sprite.Sprite):
   def __init__(self, game, x, y, speed=1, turn_speed=0.75):
       """
       Initializes the MovingSprite
       Args: 
       game (class) - all variables can be accessed through game.var
       x (int) - x position
       y (int) - y position
       speed (int) - speed of sprite
       turn_speed (float) - turning speed of the sprite
       
       Returns: None
       """
       # Init vars and image
       pygame.sprite.Sprite.__init__(self)
       self.game = game
       self.image = pygame.Surface((30, 30), pygame.SRCALPHA)
       pygame.draw.polygon(self.image, (0, 255, 255),
                         [(25, 15), (5, 5), (5, 25)])
      
       self.original_image = self.image 
       self.rect = self.image.get_rect(center=(x, y))
       self.x = x
       self.y = y
       self.position = pygame.math.Vector2(x, y)
       self.velocity = pygame.math.Vector2(speed, 0) 
       self.direction = 0 
       self.speed = speed
       self.turn_speed = turn_speed
   def update(self):
       """
       Updates the player (turning and moving)
       Args: None
       Returns: None
       """
       # Player actions and their corresponding results
       turn_efficiency = 1.5 - (self.speed / 5) * 0.5
       #Turning
       if self.game.input['left']:
           self.direction += self.turn_speed * turn_efficiency
       if self.game.input['right']:
           self.direction -= self.turn_speed * turn_efficiency
       #Changing Speed
       if self.game.input['throttle-up']:
           if self.speed <= 5:
               self.speed += 0.01
       elif self.game.input['throttle-down']:
           if self.speed >= 0.75:
               self.speed -= 0.01
      
       # Move player based on direction and speed
       self.direction %= 360
       self.velocity.from_polar((self.speed, -self.direction))
       self.position += self.velocity
       self.rect.center = self.position
       
       # Rotate image to match direction
       self.image = pygame.transform.rotate(self.original_image, self.direction)
       self.rect = self.image.get_rect(center=self.rect.center)
      
       # Out of bounds warning
       if self.rect.right < 0 or self.rect.left > SCREENX or self.rect.bottom < 0 or self.rect.top > SCREENY:
           self.game.data['warningTimer'] -= 1
           self.game.data["showWarning"] = True
       else:
           self.game.data['warningTimer'] = 480
           self.game.data["showWarning"] = False
       if self.game.data['warningTimer'] <= 0:
           self.game.data["endGame"] = True #TRUE

# Bullet sprite - Used as a group and launched by player sprite
class Bullet(pygame.sprite.Sprite):
   def __init__(self, x, y, direction, speed=10):
       """
       Initializes the Bullet sprite
       Args: 
       x (int) - x position
       y (int) - y position
       direction (float) - direction of which the bullet is pointing
       speed (int) - speed of sprite
       
       Returns: None
       """
       # init vars and image
       pygame.sprite.Sprite.__init__(self)
      
       self.image = pygame.Surface((10, 10), pygame.SRCALPHA)
       pygame.draw.rect(self.image, "yellow", (0, 0, 5, 1))
      
       self.original_image = self.image 
       self.rect = self.image.get_rect(center=(x, y))
       self.x = x
       self.y = y
       self.position = pygame.math.Vector2(x, y)
       self.direction = direction
       self.speed = speed
       self.velocity = pygame.math.Vector2()
       self.velocity.from_polar((speed, -direction))
  
   def update(self):
       """
       Updates the Bullet sprite (moving, despawning)
       Args: None
       Returns: None
       """
       # Moves the bullet every frame
       self.position += self.velocity
       self.rect.center = self.position
       
       # Despawning bullet if out of bounds
       if  self.rect.top > SCREENY:
           self.kill()
       if self.rect.bottom < 0:
           self.kill()
       if self.rect.top > SCREENY:
           self.kill()
       if self.rect.right < 0:
           self.kill()
       if self.rect.left > SCREENX:
           self.kill()
        
       #Rotates the bullet to fit where it's going
       self.image = pygame.transform.rotate(self.original_image, self.direction)

# Enemy sprite - used as a group
class Target(pygame.sprite.Sprite):
   def __init__(self, x, y, direction, speed=2):
       """
       Initializes the Target sprite
       Args: 
       x (int) - x position
       y (int) - y position
       direction (float) - direction of which the bullet is pointing
       speed (int) - speed of sprite
       
       Returns: None
       """
       pygame.sprite.Sprite.__init__(self)

       self.image = pygame.Surface((30, 30), pygame.SRCALPHA)
       pygame.draw.polygon(self.image, (255, 0, 0),
                         [(25, 15), (5, 5), (5, 25)])
       self.original_image = self.image 
       self.rect = self.image.get_rect(center=(x, y))
      
       self.position = pygame.math.Vector2(x, y)
       self.direction = direction
       self.speed = speed
       self.velocity = pygame.math.Vector2()
       self.velocity.from_polar((speed, -direction))
  
   def update(self):
       """
       Updates the Target sprite (moving, despawning)
       Args: None
       Returns: None
       """
       # Moves the target sprite
       self.position += self.velocity
       self.rect.center = self.position
       
       # Despawning if out of bounds
       if  self.rect.top > SCREENY:
           self.kill()
       if self.rect.bottom < 0:
           self.kill()
       if self.rect.top > SCREENY:
           self.kill()
       if self.rect.right < 0:
           self.kill()
       if self.rect.left > SCREENX:
           self.kill()
        
       # Rotates the image to match where it's going
       self.image = pygame.transform.rotate(self.original_image, self.direction)


# GAME CLASS - The whole game is pretty much in here
class Game:
    
    def __init__(self):
        """
        Inits game
        Args: None
        Returns: None
        """
        # Init game vars
        self.w = SCREENX
        self.h = SCREENY
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        
        # Init Player, enemies, bullets
        self.targets = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()
        self.players = pygame.sprite.Group()
        self.player = MovingSprite(self, SCREENX/2, SCREENY/2)
        
        self.players.add(self.player)
        self.reset()
        self.relativeAngle = 0
        self.lineToggle = False
        self.graphToggle = False
        
    
    def reset(self):
        """
        Resets the game - essencially only resetting what has to be reset instead of re-init-ing the whole game (more resource costly)
        Args: None
        Returns: None
        """
        self.data = {"score":0, "ammo":600, "showWarning":False, "warningTimer":480, "endGame":False}
        
        self.input = {"left":False, "right":False, "throttle-up":False, "throttle-down":False, "shoot":False}
        self.reward = 0
        self.frame_iteration = 0
   
        self.players.empty()
        self.bullets.empty()
        self.targets.empty()

        self.player = MovingSprite(self, SCREENX/2, SCREENY/2)
        self.players.add(self.player)
        
    def playStep(self, action):
        """
        One frame (update) of the game, and drawing (displaying) it
        Args: Action (list) - Action that the AI took
        Returns:
        self.reward (int) - the reward the AI receives for this frame
        self.data['endGame'] (bool) - if the game has ended or not
        self.data['score'] (int) - how many enemies did the AI kill for this game
        """
        # Init and reset somet hings
        keys = pygame.key.get_pressed()
        showAction = 'None'
        self.frame_iteration += 1
        self.reward = 0
        self.input = {"left":False, "right":False, "throttle-up":False, "throttle-down":False, "shoot":False}
        self.screen.fill("NAVYBLUE")
        
        # Calculates bullet collision with enemy (kill)
        for bullet in self.bullets:
           for target in self.targets:
               if pygame.sprite.collide_rect(bullet, target):
                   pygame.draw.circle(self.screen, 'yellow', (int(bullet.position.x), int(bullet.position.y)), 10)
                   bullet.kill()
                   target.kill()
                   self.reward += 100
                   self.data["score"] += 1
        
        # Spawns enemy if there are no more enemies to kill
        if len(self.targets) == 0:
            x, y = random.randint(0, SCREENX), random.randint(0, SCREENY)
            # x, y = SCREENX/4, SCREENY/4
            randi = random.randint(0, 360)
            randspeed = random.randint(25, 200)
            # randspeed = 0
            self.targets.add(Target(x, y, randi, randspeed/100))
        
        # Calculating relative angle (how many degrees to turn left or right to be looking at the enemy)
        enemy = self.targets.sprites()[0]

        dx = enemy.position.x - self.player.position.x
        dy = enemy.position.y - self.player.position.y
        angleE = math.degrees(math.atan2(-dy, dx)) % 360
        
        angleD = self.player.direction
        # If player is to the right of enemy then add 180 to fix the angle because of cast rule (x would be negative)
        self.relativeAngle = (angleE - angleD + 540) % 360 - 180  # Gives -180 to 180
        self.reward +=  (1-(abs(self.relativeAngle)/180))*15
        # print(self.relativeAngle)
        
        # Pygame event loop (for toggling lines and graph, also for being able to close the game)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    if self.lineToggle:
                        self.lineToggle = False
                    else:
                        self.lineToggle = True
                if event.key == pygame.K_g:
                    if self.graphToggle:
                        self.graphToggle = False
                    else:
                        self.graphToggle = True
        
        # Visual debugging - draw direction lines (on keypress cuz its cool)
        if self.lineToggle:
            pygame.draw.line(self.screen, (255,0,0),  # Red - player direction
                            self.player.position,
                            ((self.player.position.x + 50 * math.cos(math.radians(angleD))),
                            (self.player.position.y - 50 * math.sin(math.radians(angleD)))),  # Note: Negative y
                            2)

            pygame.draw.line(self.screen, (0,255,0),  # Green - to enemy
                            self.player.position,
                            self.targets.sprites()[0].position,
                            2)
        
        # print(f"Player: {angleD:.1f}° | Enemy: {angleE:.1f}° | Relative: {self.relativeAngle:.1f}° | Reward: {self.reward:.2f}")
        # Input (action)
        """
        Action Stuff:
        [1, 0, 0, 0, 0, 0] -> left
        [0, 1, 0, 0, 0, 0] -> right
        [0, 0, 1, 0, 0, 0] -> throttle up
        [0, 0, 0, 1, 0, 0] -> throttle down
        [0, 0, 0, 0, 1, 0] -> shoot
        [0, 0, 0, 0, 0, 1] -> no action taken (straight)
        
        In terms of one list...
        [left, right, throttle-up, throttle-down, shoot, no action]
        """
        
        # Regesters AI actions as an input (easier than writing np.array_equal(bla bla) every time)
        if np.array_equal(action, [1, 0, 0]):
            self.input['left'] = True
            showAction = 'Left'
        if np.array_equal(action, [0, 1, 0]):
            self.input['right'] = True
            showAction = 'Right'
        if np.array_equal(action, [0, 0, 1, 0, 0, 0]):
            self.input['throttle-up'] = True
            showAction = 'Throttle Up'
        if np.array_equal(action, [0, 0, 0, 1, 0, 0] ):
            self.input['throttle-down'] = True
            showAction = 'Throttle Down'
        if np.array_equal(action, [0, 0, 0, 0, 1, 0] ):
            self.input['shoot'] = True
            showAction = 'Shoot'
        if np.array_equal(action, [0, 0, 1]):
            self.reward -= 0.2
            
        # Allows user input for testing
        if keys[pygame.K_a]:
            self.input['left'] = True
            showAction = 'Left'
        if keys[pygame.K_d]:
            self.input['right'] = True
            showAction = 'Right'
        if keys[pygame.K_w]:
            self.input['throttle-up'] = True
            showAction = 'Throttle Up'
        if keys[pygame.K_s]:
            self.input['throttle-down'] = True
            showAction = 'Throttle Down'
        if keys[pygame.K_SPACE]:
            self.input['shoot'] = True
            showAction = 'Shoot'
        
        # Shooting - either on user input or by trigger shooting (if enemy within + or - 5 degrees it automatically shoots)
        if self.data["ammo"] > 0:
           if self.input['shoot'] or abs(self.relativeAngle) < 5 :
               self.bullets.add(Bullet(self.player.position.x, self.player.position.y, self.player.direction, self.player.speed+10))
               self.data["ammo"] -= 1
            #    self.reward -= 0.05 #Teaches ammo saving (removed, as the AI can no longer control when it shoots)
        else:
           self.data['endGame'] = True #TRUE
            
        #Too slow! AI could just be stalling time so this is to prevent it, it adds 2 minutes every time it kills a target.
        if self.frame_iteration > 14400 + (14400*self.data['score']) : 
            self.data['endGame'] = True #TRUE
        
        if self.data['warningTimer'] < 480:
            self.reward -= 10 #Teaches to stay in bounds
            
        

        
        # Update
        self.players.update()
        self.bullets.update()
        self.targets.update()
        
        # Draw
        self.players.draw(self.screen)
        self.bullets.draw(self.screen)
        self.targets.draw(self.screen)
        
        # Display texts
        aa = normalFont.render(f'FPS: {self.clock.get_fps()}', False, (255,255, 0))
        self.screen.blit(aa, (SCREENX-SCREENX/4,0))
        bb = normalFont.render(f'Action: {showAction}', False, (255,255, 0))
        self.screen.blit(bb, (SCREENX-SCREENX/4, 30))
        lines = normalFont.render(f'Lines: {self.lineToggle}', False, (255,255, 0))
        self.screen.blit(lines, (0, SCREENY-30))
        graphs = normalFont.render(f'Graphs: {self.graphToggle}', False, (255,255, 0))
        self.screen.blit(graphs, (0, SCREENY-60))
        text_surface = normalFont.render(f'Rounds Left: {self.data["ammo"]}', False, (255,255, 0))
        self.screen.blit(text_surface, (0,0))
        scoretext = normalFont.render(f'Score: {self.data["score"]}', False, (255,255, 0))
        self.screen.blit(scoretext, (0,30))
        scoretext = normalFont.render(f'Game Ended: {self.data["endGame"]}', False, (255,255, 0))
        self.screen.blit(scoretext, (0,60))
        
        # Display warning text
        if self.data["showWarning"]:
           warningtext = normalFont.render(f'RETURN TO BATTLEFIELD: {int(self.data["warningTimer"]/FPS)}', False, (255,0, 0))
           self.screen.blit(warningtext, (SCREENX/2 - warningtext.get_width()/2, SCREENY/2))
           
        # Updates (flips) the display
        pygame.display.update()
        pygame.event.pump()
        self.clock.tick() #Usually inside has FPS but i'm MAXING IT FOR RAPID TRAINING
        
        # Print why the game was ended
        if self.data['endGame']:
            print(f"Game ended because:")
            if self.data["ammo"] <= 0:
                print("- Out of ammo")
            if self.frame_iteration > 14400 * max(1, self.data['score']):
                print("- Timeout")
            if self.data['warningTimer'] <= 0:
                print("- Out of bounds too long")
                
        self.reward -= 0.5 # Avoids passive gameplay
        if self.data['endGame']:
            self.reward -= 100
        return self.reward, self.data['endGame'], self.data['score']
