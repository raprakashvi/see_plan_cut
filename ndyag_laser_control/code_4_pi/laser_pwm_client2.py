#!/usr/bin/env python3
import socket
import json
import RPi.GPIO as GPIO
import time
import threading
from typing import Optional, Any

class PWMController:
    """
    Remote PWM controller for laser operations.
    Communicates with the PWM server using JSON commands over a socket.
    """
    def __init__(self, host: str, port: int = 8000) -> None:
        self.host = host
        self.port = port

    def _send_command(self, command: dict) -> dict:
        with socket.create_connection((self.host, self.port), timeout=5) as sock:
            sock.sendall(json.dumps(command).encode('utf-8'))
            response_data = sock.recv (1024)
            if not response_data:
                raise RuntimeError("No response from PWM server.")
            return json.loads(response_data.decode('utf-8'))

    def set_pwm(self, duty_cycle: float, frequency: float) -> dict:
        """
        Set PWM duty cycle and frequency.
        """
        command = {
            'action': 'set_pwm',
            'duty_cycle': duty_cycle,
            'frequency': frequency
        }
        return self._send_command(command)

    def start(self, duty_cycle: Optional[float] = None) -> dict:
        """
        Start the PWM output. Optionally provide a duty cycle.
        """
        command = {'action': 'start'}
        if duty_cycle is not None:
            command['duty_cycle'] = duty_cycle
        return self._send_command(command)

    def stop(self) -> dict:
        """
        Stop the PWM output.
        """
        command = {'action': 'stop'}
        return self._send_command(command)

    def status(self) -> dict:
        """
        Get the current PWM status.
        """
        command = {'action': 'status'}
        return self._send_command(command)


def integrate_laser_control(controller: PWMController, command: str,
                            duty_cycle: Optional[float] = None,
                            frequency: Optional[float] = None) -> Any:
    """
    Integration helper for remote laser control.

    Parameters:
      controller: PWMController instance.
      command: One of "start", "stop", "set_pwm", or "status".
      duty_cycle: For "start" (optional) or required for "set_pwm".
      frequency: Required for "set_pwm".

    Returns:
      The PWM server's response.
    """
    if frequency is not None and frequency > 290:
        raise ValueError("Frequency must be less than or equal to 290 (max 300) Hz.")
        print("Frequency must be less than or equal to 300 Hz.")

    if duty_cycle is not None and (duty_cycle < 0 or duty_cycle > 99):
        raise ValueError("Duty cycle must be between 0 and 100.")
        print("Duty cycle must be between 0 and 100.")

    if command == "start":
        return controller.start(duty_cycle)
    elif command == "stop":
        return controller.stop()
    elif command == "set_pwm":
        if duty_cycle is None or frequency is None:
            raise ValueError("Both duty_cycle and frequency must be provided for 'set_pwm'.")
        return controller.set_pwm(duty_cycle, frequency)
    elif command == "status":
        return controller.status()
    else:
        raise ValueError("Invalid command. Use 'start', 'stop', 'set_pwm', or 'status'.")

# Button press handler
class ButtonHandler:
    """Button handler with threading and event-based control"""
    def __init__(self, pin, pwm_controller, default_duty_cycle=10, default_frequency=250):
        self.pin = pin
        self.pwm_controller = pwm_controller
        self.default_duty_cycle = default_duty_cycle
        self.default_frequency = default_frequency
        self.press_count = 0
        self.last_press_time = 0
        self.button_event = threading.Event()
        self.running = True
        
        # Initialize GPIO for button
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(pin, GPIO.FALLING, callback=self._button_pressed, bouncetime=300)
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_button_events)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def _button_pressed(self, channel):
        """Button press callback - sets the event"""
        current_time = time.time()
        # Basic software debounce in addition to hardware
        if (current_time - self.last_press_time) > 0.3:
            self.last_press_time = current_time
            self.button_event.set()
    
    def _process_button_events(self):
        """Thread to process button events"""
        while self.running:
            # Wait for button event
            if self.button_event.wait(0.1):  # Check every 100ms
                self.press_count += 1
                print(f"Button press #{self.press_count} detected")
                
                try:
                    if self.press_count % 2 == 1:  # Odd press - turn on laser
                        print(f"Odd press: Turning laser ON with duty cycle {self.default_duty_cycle}%")
                        integrate_laser_control(self.pwm_controller, "start", duty_cycle=self.default_duty_cycle)
                        integrate_laser_control(self.pwm_controller, "set_pwm", 
                                             duty_cycle=self.default_duty_cycle, 
                                             frequency=self.default_frequency)
                    else:  # Even press - turn off laser
                        print("Even press: Turning laser OFF (duty cycle = 0)")
                        integrate_laser_control(self.pwm_controller, "set_pwm", duty_cycle=0, frequency=self.default_frequency)
                except Exception as e:
                    print(f"Error handling button press: {e}")
                
                # Reset event for next press
                self.button_event.clear()
    
    def stop(self):
        """Stop the button handler"""
        self.running = False
        self.button_event.set()  # Ensure thread exits wait
        self.process_thread.join(1.0)  # Wait up to 1 second for thread to finish
        GPIO.remove_event_detect(self.pin)


# Example usage:
if __name__ == "__main__":
    # Replace '192.168.1.100' with the PWM server's IP address.
    pwm = PWMController("10.194.210.35")
    
    try:
        # Create button handler using GPIO 18
        button_handler = ButtonHandler(pin=18, pwm_controller=pwm, 
                                       default_duty_cycle=10, default_frequency=250)
        
        print("Button handler initialized on GPIO 18.")
        print("Press button an odd number of times to turn laser ON.")
        print("Press button an even number of times to turn laser OFF.")
        print("Press Ctrl+C to exit.")
        
        # Start the PWM with an optional duty cycle (e.g., 10%).
        response = integrate_laser_control(pwm, "start", duty_cycle=10)
        print("Start response:", response)
        
        # Set the PWM to 50% duty cycle at 1000Hz.
        response = integrate_laser_control(pwm, "set_pwm", duty_cycle=50, frequency=250)
        print("Set PWM response:", response)
        
        # Retrieve the current status.
        response = integrate_laser_control(pwm, "status")
        print("Status response:", response)
        
        # Stop the PWM output.
        response = integrate_laser_control(pwm, "stop")
        print("Stop response:", response)
        
        # Keep the program running
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nExiting program")
    finally:
        # Clean up resources
        if 'button_handler' in locals():
            button_handler.stop()
        GPIO.cleanup()