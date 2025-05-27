from robotiq_gripper_python import RobotiqGripper
import time
import serial

ser = serial.Serial("/dev/ttyUSBGripper", 115200, timeout=0.2)

if __name__ == "__main__":
    gripper = RobotiqGripper(comport="/dev/ttyUSBGripper")

    gripper.start()

    for i in range(3):
        gripper.move(pos=255, vel=255, force=255, block=True)
        gripper.move(pos=0, vel=255, force=255, block=True)

    for i in range(255):
        gripper.move(pos=i, vel=255, force=255, block=False)
        time.sleep(0.01)

    gripper.shutdown()