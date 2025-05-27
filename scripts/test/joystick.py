import pygame   
from hsl_ur5.input.joystick import Joystick

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