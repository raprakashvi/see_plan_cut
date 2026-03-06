import RPi.GPIO as GPIO
from time import sleep
import numpy as np

class LaserPWM():
    def __init__(self, laser_pin=12, freq_initial=250, dutycycle_initial=10):
        ## TODO: Add checks to make sure freq does not cross 290 and dutycycle does not cross 100
        
        self.laser_pin = laser_pin
        self.freq = freq_initial
        self.duty_cycle = dutycycle_initial

        # Setup
        GPIO.setwarnings(False)#disable warnings
        GPIO.setmode(GPIO.BCM)#pin numbering system 
        #BCM uses actual GPIO numbers instead of physical pin position(BOARD)
        GPIO.setup(self.laserpin,GPIO.OUT)

    
    def laser_initial_start(self,duty_cycle=5):
        """Start the laser at low duty cycle and wait for stability
        Args: duty_cycle = initial low duty cycle for the laser
        return NONE
        """
        self.pi_pwm = GPIO.PWM(self.laserpin, self.freq) #create PWM instance with frequency
        self.pi_pwm.start(duty_cycle)

    def laser_control(self,freq_initial=250, dutycycle_initial=10):
        """Sets the duty cycle and frequency of the laser based on parameters passed.
        Args: freq_initial = Lower than 300 Hz
        dutycycle_initial = lower than 100 %
        """

        print("Laser control about to start")
        print(f"Duty cycle is going to be {dutycycle_initial}.")
        set_dtcycle = dutycycle_initial
        
        if set_dtcycle != self.duty_cycle:
            dutycycle = set_dtcycle
            self.pi_pwm.ChangeDutyCycle(dutycycle)

        stop = input('press any key to stop ')
        # any key will stop 
        self.pi_pwm.stop()
        GPIO.cleanup()
      
    def laser_control_user(self,freq_initial=250, dutycycle_initial=10):
        """Sets the duty cycle and frequency of the laser based on user requirement
        Args: freq_initial = Lower than 300 Hz
        dutycycle_initial = lower than 100 %
         
        TODO: double check if arguments are needed or not in this.
        """
        print("Laser control about to start")
        set_dtcycle = float(input("Enter value between 5 to 99 for laser dutycycle "))
        
        if set_dtcycle != self.duty_cycle:
            dutycycle = set_dtcycle
            self.pi_pwm.ChangeDutyCycle(dutycycle)
       
        stop = input('press any key to stop ')
        # any key will stop 
        self.pi_pwm.stop()
        GPIO.cleanup()

if __name__=="__main__":
    # unit test
    freq_initial = 250 
    dutycycle_initial = 5
    laser = LaserPWM(freq_initial=freq_initial , duty_cycle=dutycycle_initial)
    
    # test 1: check laser start
    laser.laser_initial_start()

    # test 2: check laser control user
    laser.laser_control_user()
