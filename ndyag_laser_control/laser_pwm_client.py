#!/usr/bin/env python3
import socket
import json
from typing import Optional, Any
import time

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
            response_data = sock.recv(1024)
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

    if duty_cycle is not None and (duty_cycle < 0 or duty_cycle > 100):
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


# Example usage:
if __name__ == "__main__":
    # Replace '192.168.1.100' with the PWM server's IP address.
    pwm = PWMController("10.194.210.35")
        
    # Parameters
    time_period = 1.0 # Adjust the time period as needed.
    frequency = 100  # Frequency in Hz, must be less than or equal to 290 Hz.
    duty_cycle = 100  # Initial duty cycle, can be adjusted as needed.


    
    # Start the PWM with an optional duty cycle (e.g., 10%).
    response = integrate_laser_control(pwm, "start", duty_cycle=0)
    print("Start response:", response)

    input("Press Enter to continue...")  # Keep the console open to see the output before proceeding.
    start_time = time.time()
    while (time.time() - start_time) < time_period:
        response = integrate_laser_control(pwm, "set_pwm", duty_cycle=duty_cycle, frequency=frequency)
        print("Set PWM response:", response)

     # Stop the PWM output.
    response = integrate_laser_control(pwm, "stop")
    print("Stop response:", response)

    # Retrieve the current status.
    response = integrate_laser_control(pwm, "status")

    print("Status response:", response)
    
    # input("Press Enter to continue...")  # Keep the console open to see the output before stopping.

   
