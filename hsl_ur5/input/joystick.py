import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

MAX_GRIPPER_DELTA = 0.1
MAX_TRANSLATION = 0.12
MAX_ROTATION = 0.1
MAX_Z_ROTATION = 0.4

class Joystick:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        # Initialize the joystick module
        pygame.joystick.init()

        # Assuming only one joystick is connected
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.active = False

        self.axis_states = {}
        self.button_states = {}
        self.hat_states = {}

    def get_button_state(self):
        # Get the state of all buttons
        button_states = {}
        for i in range(self.joystick.get_numbuttons()):
            val = self.joystick.get_button(i)

            if val == 1:
                self.active = True

            button_states[f'button_{i}'] = val

        return button_states

    def get_axis_state(self):
        # Get the state of all axes
        axis_states = {}
        for i in range(self.joystick.get_numaxes()):
            val = self.joystick.get_axis(i)

            if abs(val) > 1e-4:
                self.active = True

            axis_states[f'axis_{i}'] = val

        if not self.active:
            axis_states['axis_4'] = -1
            axis_states['axis_5'] = -1


        return axis_states
    
    def rumble(self, val=1.0, duration=1.0):
        """
        Rumble the controller
        param val: The strength of the rumble (0 to 1)
        """
        self.joystick.rumble(0, val, int(duration * 1000))

    def stop_rumble(self):
        self.joystick.stop_rumble()

    def get_hat_state(self):
        # Get the state of all hats (D-pad)
        hat_states = {}
        for i in range(self.joystick.get_numhats()):
            hat_states[f'hat_{i}'] = self.joystick.get_hat(i)
        return hat_states

    def update(self):
        # Process pygame events
        pygame.event.pump()
        self.axis_states = self.get_axis_state()
        self.button_states = self.get_button_state()
        self.hat_states = self.get_hat_state()

    def close(self):
        # Quit pygame
        pygame.quit()

    def get_twist(self, max_translation=MAX_TRANSLATION, max_rotation=MAX_ROTATION, max_z_rotation=MAX_Z_ROTATION): 
        if len(self.axis_states) == 0:
            axis_state = self.get_axis_state()
        else:
            axis_state = self.axis_states

        if len(self.button_states) == 0:
            button_state = self.get_button_state()
        else:
            button_state = self.button_states

        twist = np.zeros(6, dtype=np.float32)
        # left and right
        if abs(axis_state['axis_0']) > 0.15:
            twist[0] += axis_state['axis_0'] * max_translation
        # forward and backward
        if abs(axis_state['axis_1']) > 0.15:
            twist[1] += -axis_state['axis_1'] * max_translation
        # up and down
        if axis_state['axis_2'] > -0.9:
            twist[2] += -(axis_state['axis_2']+1)/2 * max_translation
        if axis_state['axis_5'] > -0.9:
            twist[2] += (axis_state['axis_5']+1)/2 * max_translation
        # rotate left and right
        if abs(axis_state['axis_4']) > 0.2:
            twist[3] += axis_state['axis_4'] * max_rotation
        if abs(axis_state['axis_3']) > 0.2:
            twist[4] += axis_state['axis_3'] * max_rotation
        # rotate yaw
        if button_state['button_2'] == 1:
            twist[5] += max_z_rotation 
        if button_state['button_3'] == 1:
            twist[5] -= max_z_rotation 

        return twist
    
    def get_gripper_delta(self, max_gripper_delta=MAX_GRIPPER_DELTA):

        if len(self.button_states) == 0:
            button_state = self.get_button_state()
        else:
            button_state = self.button_states

        button_state = self.get_button_state()
        if button_state['button_0'] == 1:
            return max_gripper_delta
        if button_state['button_1'] == 1:
            return -max_gripper_delta
        return 0.0
    
    def is_start_pressed(self):
        if len(self.button_states) == 0:
            button_state = self.get_button_state()
        else:
            button_state = self.button_states
        return button_state['button_7'] == 1
    
    def is_back_pressed(self):
        if len(self.button_states) == 0:
            button_state = self.get_button_state()
        else:
            button_state = self.button_states
        return button_state['button_6'] == 1
    
# Code to test the XboxController class
if __name__ == "__main__":
    controller = Joystick()
    
    try:
        while True:
            controller.update()
            print("Twist:", controller.get_twist())
            print("Gripper Delta:", controller.get_gripper_delta())
            print("Button States:", controller.button_states)

            # controller.rumble(val=0.5)

            # Add a small delay to prevent flooding the console with prints
            pygame.time.wait(100)
    except KeyboardInterrupt:
        # Exit the loop on Ctrl+C
        pass
    finally:
        controller.close()