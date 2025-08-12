from ddsm115 import MotorControl
import matplotlib.pyplot as plt
import time
import math
import csv
import numpy as np
from datetime import datetime

filename = f"trajectory_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
WHEEL_RADIUS = 0.05  # meter
WHEEL_BASE = 0.278  # meter

def rpm_to_mps(rpm):
    return (rpm / 60.0) * 2 * math.pi * WHEEL_RADIUS

def calculate_rmse(actual, reference):
    """Calculate RMSE between actual and reference trajectories"""
    mask = ~(np.isnan(actual) | np.isnan(reference))
    actual_clean = actual[mask]
    reference_clean = reference[mask]

    if len(actual_clean) == 0:
        return 0.0

    rmse = np.sqrt(np.mean((actual_clean - reference_clean) ** 2))
    return rmse

class RobotStraightPath:
    def __init__(self, port="COM6"):
        self.mc = MotorControl(device=port)
        self.mc.set_drive_mode(_id=1, _mode=2)
        self.mc.set_drive_mode(_id=2, _mode=2)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # Movement state tracking
        self.movement_mode = "stopped"  # "forward", "rotating", "stopped"
        self.locked_theta = 0.0  # Theta yang dikunci untuk gerakan lurus
        self.target_theta = 0.0  # Target theta untuk rotasi
        
        # Pembatasan theta - Parameter yang bisa disesuaikan
        self.theta_tolerance = math.radians(1.0)  # Toleransi 1 derajat
        self.theta_lock_strength = 0.95  # Kekuatan penguncian theta (0-1)
        self.theta_correction_gain = 0.6  # Gain untuk koreksi theta
        
        # Precise timing control
        self.target_interval = 0.0  # 20ms target interval
        self.start_time = time.perf_counter()
        self.prev_time = self.start_time
        self.next_log_time = self.start_time

        self.xs = []
        self.ys = []
        self.theta_history = []
        self.data_log = []

        # RMSE tracking
        self.rmse_x_values = []
        self.rmse_y_values = []
        self.rmse_total_values = []

        # Setup plotting
        plt.ion()
        self.fig, ((self.ax1, self.ax2)) = plt.subplots(1, 2, figsize=(15, 12))

        # Trajectory plot
        (self.line,) = self.ax1.plot([], [], "-b", lw=2, label="Actual Path")
        # (self.ideal_line,) = self.ax1.plot([], [], "-g", lw=1, alpha=0.7, label="Ideal Path")
        self.draw_square()
        self.ax1.set_xlabel("X (cm)")
        self.ax1.set_ylabel("Y (cm)")
        self.ax1.set_title("Robot Trajectory with Theta Constraint")
        self.ax1.grid(True)
        self.ax1.set_xlim(-50, 125)
        self.ax1.set_ylim(-50, 125)
        self.ax1.legend()

        # RMSE plot
        (self.rmse_line,) = self.ax2.plot([], [], "-r", lw=2, label="RMSE")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("RMSE (cm)")
        self.ax2.set_title("Real-time RMSE")
        self.ax2.grid(True)
        self.ax2.legend()

        # Theta tracking plot
        # (self.theta_line,) = self.ax3.plot([], [], "-m", lw=2, label="Actual Theta")
        # (self.theta_locked_line,) = self.ax3.plot([], [], "--c", lw=2, label="Locked Theta")
        # self.ax3.set_xlabel("Time (s)")
        # self.ax3.set_ylabel("Theta (degrees)")
        # self.ax3.set_title("Theta Control")
        # self.ax3.grid(True)
        # self.ax3.legend()

        # Trajectory error plot
        # (self.error_line,) = self.ax4.plot([], [], "-m", lw=2, label="Position Error")
        # self.ax4.set_xlabel("Time (s)")
        # self.ax4.set_ylabel("Error (cm)")
        # self.ax4.set_title("Trajectory Error from Ideal")
        # self.ax4.grid(True)
        # self.ax4.legend()

        # Ideal path tracking
        # self.ideal_xs = []
        # self.ideal_ys = []
        self.position_errors = []

    def draw_square(self, side_cm=100):
        square_x = [0, side_cm, side_cm, 0, 0]
        square_y = [0, 0, side_cm, side_cm, 0]
        self.ax1.plot(square_x, square_y, "r--", linewidth=2, label="Target Square")

    def constrain_theta_advanced(self, raw_theta):
        """Pembatasan theta yang lebih canggih berdasarkan mode gerakan"""
        if self.movement_mode == "forward":
            # Mode gerakan maju - theta dikunci ketat
            theta_error = raw_theta - self.locked_theta
            
            # Normalisasi error ke range [-pi, pi]
            theta_error = (theta_error + math.pi) % (2 * math.pi) - math.pi
            
            # Terapkan dead zone dan pembatasan
            if abs(theta_error) < self.theta_tolerance:
                # Dalam toleransi - gunakan interpolasi dengan locked theta
                return self.locked_theta * self.theta_lock_strength + raw_theta * (1 - self.theta_lock_strength)
            else:
                # Di luar toleransi - batasi perubahan
                max_correction = self.theta_tolerance * self.theta_correction_gain
                corrected_error = math.copysign(max_correction, theta_error)
                return self.locked_theta + corrected_error
                
        elif self.movement_mode == "rotating":
            # Mode rotasi - biarkan theta berubah bebas
            return raw_theta
            
        else:
            # Mode stopped - maintain current theta
            return self.theta

    # def update_ideal_path(self):
    #     """Update jalur ideal berdasarkan gerakan yang direncanakan"""
    #     if self.movement_mode == "forward":
    #         # Jalur lurus dari posisi awal ke target
    #         if len(self.ideal_xs) == 0:
    #             self.ideal_xs.append(self.x * 100)
    #             self.ideal_ys.append(self.y * 100)
    #         else:
    #             # Tambahkan titik ideal berdasarkan locked theta
    #             last_ideal_x = self.ideal_xs[-1] / 100
    #             last_ideal_y = self.ideal_ys[-1] / 100
                
    #             # Hitung pergerakan ideal
    #             dt = self.target_interval
    #             v_ideal = 0.2  # kecepatan ideal (m/s)
                
    #             ideal_x = last_ideal_x + v_ideal * math.cos(self.locked_theta) * dt
    #             ideal_y = last_ideal_y + v_ideal * math.sin(self.locked_theta) * dt
                
    #             self.ideal_xs.append(ideal_x * 100)
    #             self.ideal_ys.append(ideal_y * 100)

    def calculate_position_error(self):
        # """Hitung error posisi dari jalur ideal"""
        # if len(self.ideal_xs) > 0 and len(self.xs) > 0:
        #     # Ambil posisi terakhir
        #     actual_x = self.xs[-1]
        #     actual_y = self.ys[-1]
        #     ideal_x = self.ideal_xs[-1] if len(self.ideal_xs) <= len(self.xs) else self.ideal_xs[len(self.xs)-1]
        #     ideal_y = self.ideal_ys[-1] if len(self.ideal_ys) <= len(self.ys) else self.ideal_ys[len(self.ys)-1]
            
        #     error = math.sqrt((actual_x - ideal_x)**2 + (actual_y - ideal_y)**2)
        #     return error
        # return 0.0
        if len(self.xs) > 1:
            # Hitung deviasi dari jalur lurus untuk gerakan forward
            if self.movement_mode == "forward":
                # Hitung jarak perpendicular dari garis lurus
                dx = self.xs[-1] - self.xs[0]
                dy = self.ys[-1] - self.ys[0]
                distance = math.sqrt(dx**2 + dy**2)
                return distance * 0.01  # Simple error metric
            else:
                return 0.0
        return 0.0

    def calculate_current_rmse(self):
        """Calculate RMSE using multiple methods"""
        if len(self.xs) < 10:
            return 0.0, 0.0, 0.0

        try:
            # Method: Compare with smoothed trajectory
            window_size = min(7, len(self.xs))
            xs_smooth = np.convolve(self.xs, np.ones(window_size) / window_size, mode="same")
            ys_smooth = np.convolve(self.ys, np.ones(window_size) / window_size, mode="same")

            rmse_x = calculate_rmse(np.array(self.xs), xs_smooth)
            rmse_y = calculate_rmse(np.array(self.ys), ys_smooth)
            rmse_total = np.sqrt(rmse_x**2 + rmse_y**2)

            return rmse_x, rmse_y, rmse_total

        except Exception as e:
            print(f"RMSE calculation error: {e}")
            return 0.0, 0.0, 0.0

    def read_and_log_data_precise(self):
        """Read sensor data with theta constraint"""
        current_time = time.perf_counter()
        
        try:
            fb_rpm_1 = self.mc.get_motor_feedback(_id=1)[0]
            fb_rpm_2 = self.mc.get_motor_feedback(_id=2)[0]
        except:
            fb_rpm_1, fb_rpm_2 = 0, 0
        
        # Update pose calculation
        dt = current_time - self.prev_time
        self.prev_time = current_time

        v_l = rpm_to_mps(-fb_rpm_1)
        v_r = rpm_to_mps(fb_rpm_2)

        v = (v_r + v_l) / 2.0
        omega = (v_r - v_l) / WHEEL_BASE

        # Hitung theta mentah (tanpa pembatasan)
        raw_theta = self.theta + omega * dt
        raw_theta = (raw_theta + math.pi) % (2 * math.pi) - math.pi

        # Terapkan pembatasan theta
        constrained_theta = self.constrain_theta_advanced(raw_theta)

        # Update posisi berdasarkan mode gerakan
        if self.movement_mode == "forward":
            # Gunakan locked theta untuk perhitungan posisi
            self.x += v * math.sin(self.locked_theta) * dt
            self.y += v * math.cos(self.locked_theta) * dt
            self.theta = constrained_theta
        elif self.movement_mode == "rotating":
            # Untuk rotasi, minimal gerakan maju dan update theta normal
            self.x += v * math.sin(self.theta) * dt * 0.05  # Minimal drift
            self.y += v * math.cos(self.theta) * dt * 0.05
            self.theta = raw_theta  # Gunakan theta mentah untuk rotasi
        else:
            # Default behavior
            self.x += v * math.sin(self.theta) * dt
            self.y += v * math.cos(self.theta) * dt
            self.theta = constrained_theta

        # Convert to cm
        x_cm = self.x * 100
        y_cm = self.y * 100
        
        # Update arrays
        self.xs.append(x_cm)
        self.ys.append(y_cm)
        self.theta_history.append(math.degrees(self.theta))

        # Update ideal path
        # self.update_ideal_path()

        # Calculate errors
        pos_error = self.calculate_position_error()
        self.position_errors.append(pos_error)

        # Calculate RMSE
        rmse_x, rmse_y, rmse_total = self.calculate_current_rmse()
        self.rmse_x_values.append(rmse_x)
        self.rmse_y_values.append(rmse_y)
        self.rmse_total_values.append(rmse_total)

        # Log data at precise intervals
        if current_time >= self.next_log_time:
            elapsed_time = current_time - self.start_time
            self.data_log.append([
                current_time, x_cm, y_cm, math.degrees(self.theta),
                math.degrees(self.locked_theta), math.degrees(raw_theta),
                rmse_x, rmse_y, rmse_total, pos_error,
                dt, elapsed_time, fb_rpm_1, fb_rpm_2, self.movement_mode
            ])
            
            self.next_log_time += self.target_interval

        return fb_rpm_1, fb_rpm_2, current_time

    def precise_timing_loop(self, condition_func, action_func=None):
        """Main precise timing loop with theta constraint"""
        loop_start_time = time.perf_counter()
        next_iteration_time = loop_start_time + self.target_interval
        iteration_count = 0
        
        while condition_func():
            iteration_start = time.perf_counter()
            
            fb_rpm_1, fb_rpm_2, data_time = self.read_and_log_data_precise()
            
            if action_func:
                action_func(fb_rpm_1, fb_rpm_2, data_time)
            
            current_time = time.perf_counter()
            sleep_time = next_iteration_time - current_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            next_iteration_time += self.target_interval
            iteration_count += 1

    def update_plots_comprehensive(self):
        """Update all plots with comprehensive information"""
        if len(self.xs) > 0:
            # Trajectory plot
            self.line.set_data(self.xs, self.ys)
            # if len(self.ideal_xs) > 0:
            #     self.ideal_line.set_data(self.ideal_xs, self.ideal_ys)

            # RMSE plot
            if len(self.rmse_total_values) > 0:
                time_points = np.arange(len(self.rmse_total_values)) * self.target_interval
                self.rmse_line.set_data(time_points, self.rmse_total_values)
                self.ax2.set_xlim(0, max(1, time_points[-1]))
                self.ax2.set_ylim(0, max(1, max(self.rmse_total_values) * 1.1))

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def forward(self, distance_cm, rpm=20):
        """Gerakan maju dengan pembatasan theta"""
        distance_m = distance_cm / 100.0
        start_x = self.x
        start_y = self.y
        
        # Set mode dan lock theta
        self.movement_mode = "forward"
        self.locked_theta = self.theta  # Kunci theta saat ini
        
        print(f"\nMoving forward {distance_cm} cm at {rpm} RPM...")
        print(f"Locked theta: {math.degrees(self.locked_theta):.2f}°")
        
        self.mc.send_rpm(_id=1, rpm=-rpm)
        self.mc.send_rpm(_id=2, rpm=rpm)

        self.traveled = 0.0
        plot_counter = 0
        
        def check_distance():
            dx = self.x - start_x
            dy = self.y - start_y
            self.traveled = math.sqrt(dx**2 + dy**2)
            return self.traveled < distance_m
        
        def movement_action(fb_rpm_1, fb_rpm_2, current_time):
            nonlocal plot_counter
            plot_counter += 1
            
            if plot_counter % 10 == 0:  # Update plots more frequently
                self.update_plots_comprehensive()
            
            if plot_counter % 25 == 0:
                elapsed_time = current_time - self.start_time
                theta_error = math.degrees(self.theta - self.locked_theta)
                pos_error = self.position_errors[-1] if self.position_errors else 0
                print(
                    f"t={elapsed_time:6.2f}s | "
                    f"x={self.x*100:7.1f} y={self.y*100:7.1f} θ={math.degrees(self.theta):+6.1f}° | "
                    f"θ_err={theta_error:+5.1f}° pos_err={pos_error:5.1f}cm",
                    end="\r"
                )
        
        self.precise_timing_loop(check_distance, movement_action)
        self.stop()
        
        # Reset mode
        self.movement_mode = "stopped"
        print(f"\nForward movement completed. Distance: {self.traveled:.3f}m")

    def rotate_cw(self, degree, rpm=5):
        """Rotasi dengan kontrol theta"""
        target_deg = abs(degree)
        start_theta_deg = math.degrees(self.theta)
        
        # Set mode rotasi
        self.movement_mode = "rotating"
        self.target_theta = self.theta + math.radians(degree)
        
        print(f"\nRotating clockwise {degree}° at {rpm} RPM...")
        
        self.mc.send_rpm(_id=1, rpm=rpm)
        self.mc.send_rpm(_id=2, rpm=rpm)

        self.delta_theta = 0.0
        plot_counter = 0
        
        def check_rotation():
            current_theta_deg = math.degrees(self.theta)
            self.delta_theta = (current_theta_deg - start_theta_deg) % 360
            
            # Debug print setiap beberapa iterasi
            if plot_counter % 50 == 0:
                print(f"Debug: current={current_theta_deg:.1f}°, start={start_theta_deg:.1f}°, delta={self.delta_theta:.1f}°, target={target_deg:.1f}°")
            
            return self.delta_theta < target_deg
        
        def rotation_action(fb_rpm_1, fb_rpm_2, current_time):
            nonlocal plot_counter
            plot_counter += 1
            
            if plot_counter % 10 == 0:
                self.update_plots_comprehensive()
            
            if plot_counter % 25 == 0:
                elapsed_time = current_time - self.start_time
                print(
                    f"t={elapsed_time:6.2f}s | "
                    f"θ={math.degrees(self.theta):+6.1f}° | "
                    f"Rotated: {self.delta_theta:.1f}°",
                    end="\r"
                )
        
        self.precise_timing_loop(check_rotation, rotation_action)
        self.stop()
        
        # Update locked theta untuk gerakan selanjutnya
        self.locked_theta = self.theta
        self.movement_mode = "stopped"
        print(f"\nRotation completed. New theta: {math.degrees(self.theta):.1f}°")

    def stop(self):
        self.mc.send_rpm(_id=1, rpm=0)
        self.mc.send_rpm(_id=2, rpm=0)

    def print_final_analysis(self):
        """Print comprehensive analysis"""
        if len(self.rmse_total_values) > 0:
            final_rmse = self.rmse_total_values[-1]
            avg_rmse = np.mean(self.rmse_total_values)
            final_pos_error = self.position_errors[-1] if self.position_errors else 0
            avg_pos_error = np.mean(self.position_errors) if self.position_errors else 0
            
            print(f"\n{'=' * 70}")
            print("TRAJECTORY ANALYSIS WITH THETA CONSTRAINT")
            print(f"{'=' * 70}")
            print(f"Theta constraint parameters:")
            print(f"  - Tolerance: {math.degrees(self.theta_tolerance):.1f}°")
            print(f"  - Lock strength: {self.theta_lock_strength:.2f}")
            print(f"  - Correction gain: {self.theta_correction_gain:.2f}")
            print(f"\nTrajectory quality:")
            print(f"  - Final RMSE: {final_rmse:.3f} cm")
            print(f"  - Average RMSE: {avg_rmse:.3f} cm")
            print(f"  - Final position error: {final_pos_error:.3f} cm")
            print(f"  - Average position error: {avg_pos_error:.3f} cm")
            print(f"  - Total data points: {len(self.data_log)}")
            print(f"{'=' * 70}")

    def close(self):
        self.print_final_analysis()
        self.mc.close()
        plt.ioff()
        plt.show()


def main():
    robot = RobotStraightPath(port="COM6")

    try:
        print("Starting robot trajectory with theta constraint...")
        
        # Tes dengan satu segmen dulu
        # robot.forward(80)
        # time.sleep(1)
        robot.rotate_cw(90)
        time.sleep(1)
        # robot.forward(100)
        # time.sleep(1)
        # robot.rotate_cw(90)
        # time.sleep(1)
        # robot.forward(100)
        # time.sleep(1)
        # robot.rotate_cw(90)
        # time.sleep(1)
        # robot.forward(100)
        # time.sleep(1)
        # robot.rotate_cw(90)
        # time.sleep(1)
        
        # Uncomment untuk trajectory persegi penuh
        # robot.forward(200)
        # time.sleep(1)
        # robot.rotate_cw(90)
        # time.sleep(1)
        # robot.forward(200)
        # time.sleep(1)
        # robot.rotate_cw(90)
        # time.sleep(1)
        # robot.forward(200)
        # time.sleep(1)
        # robot.rotate_cw(90)

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt. Stopping...")

    finally:
        # Save enhanced data
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "x_cm", "y_cm", "theta_deg", "locked_theta_deg", "raw_theta_deg",
                "rmse_x", "rmse_y", "rmse_total", "position_error",
                "dt", "elapsed_time", "fb_rpm_1", "fb_rpm_2", "movement_mode"
            ])
            writer.writerows(robot.data_log)
        print(f"\nEnhanced trajectory data saved to {filename}")

        png_filename = filename.rsplit(".", 1)[0] + "_constrained.png"
        robot.fig.savefig(png_filename, dpi=300, bbox_inches="tight")
        print(f"Trajectory analysis plot saved to {png_filename}")

        robot.close()


if __name__ == "__main__":
    main()