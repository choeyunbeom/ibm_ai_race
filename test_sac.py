import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import sys
import time
import math
import os
import pandas as pd
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise

# --- Utilities ---
PI = 3.14159265359
data_size = 2**17

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def destringify(s):
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            return s
    elif type(s) is list:
        if len(s) < 2: return destringify(s[0])
        else: return [destringify(i) for i in s]

class ServerState():
    def __init__(self):
        self.d = dict()
    def parse_server_str(self, server_string):
        server_string = server_string.strip()[:-1]
        sslisted = server_string.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = destringify(w[1:])

class DriverAction():
    def __init__(self):
        self.d = { 'accel':0.2, 'brake':0, 'clutch':0, 'gear':1, 'steer':0, 'focus':[-90,-45,0,45,90], 'meta':0 }
    def __repr__(self):
        self.d['steer'] = clip(self.d['steer'], -1, 1)
        self.d['brake'] = clip(self.d['brake'], 0, 1)
        self.d['accel'] = clip(self.d['accel'], 0, 1)
        self.d['clutch'] = clip(self.d['clutch'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]: self.d['gear'] = 0
        out = str()
        for k in self.d:
            out += '('+k+' '
            v = self.d[k]
            if not type(v) is list: out += '%.3f' % v
            else: out += ' '.join([str(x) for x in v])
            out += ')'
        return out

# --- Pandas Logger Class ---
class PandasLogger:
    def __init__(self, filename="torcs_telemetry.csv"):
        self.filename = filename
        self.columns = [
            "Iteration", "Time",
            "Speed_X(km/h)", "Target_Speed_Est", 
            "Steer_Angle", "Steer_Gain_Est",     
            "Track_Pos", "Centering_Gain_Est",   
            "Brake", "Brake_Threshold_Est",      
            "Gear", "RPM", "Wheel_Spin"          
        ]
        
        # Check if file exists to determine if we need to write headers
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.filename, index=False)

    def log_step(self, iteration, state, action):
        # Extract Data
        sp_x = state.get('speedX', 0.0)
        track_pos = state.get('trackPos', 0.0)
        rpm = state.get('rpm', 0.0)
        gear = action.d['gear']
        steer = action.d['steer']
        accel = action.d['accel']
        brake = action.d['brake']
        
        # Calculate wheel spin
        wheel_spin_vel = state.get('wheelSpinVel', [0,0,0,0])
        avg_wheel_spin = sum(wheel_spin_vel) / 4.0 if wheel_spin_vel else 0.0

        # Mapping to Parameters
        target_speed = sp_x 
        steer_gain = abs(steer) 
        centering_gain = abs(steer) * abs(track_pos)
        brake_threshold = brake

        # Create a dictionary for the new row
        new_data = {
            "Iteration": iteration,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Speed_X(km/h)": round(sp_x, 2),
            "Target_Speed_Est": round(target_speed, 2),
            "Steer_Angle": round(steer, 4),
            "Steer_Gain_Est": round(steer_gain, 4),
            "Track_Pos": round(track_pos, 4),
            "Centering_Gain_Est": round(centering_gain, 4),
            "Brake": round(brake, 4),
            "Brake_Threshold_Est": round(brake_threshold, 4),
            "Gear": gear,
            "RPM": round(rpm, 0),
            "Wheel_Spin": round(avg_wheel_spin, 2)
        }

        # Convert to DataFrame (single row)
        df_row = pd.DataFrame([new_data])

        # Append to CSV using Pandas
        df_row.to_csv(self.filename, mode='a', header=False, index=False)
        
        return new_data

# --- GYM ENVIRONMENT ---
class TorcsEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, host='localhost', port=3001, sid='SCR'):
        super(TorcsEnv, self).__init__()
        
        self.host = host
        self.port = port
        self.sid = sid
        
        # Action space: [steer, accel, brake] - continuous actions
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )

        # Observation space: track sensors (19) + other features (10)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(29,), 
            dtype=np.float32
        )

        self.so = None
        self.S = ServerState()
        self.R = DriverAction()
        self.time_step = 0
        self.stuck_counter = 0
        
        # Initialize Pandas Logger
        self.logger = PandasLogger("torcs_training_log.csv")

    def step(self, action):
        # Set default meta value
        self.R.d['meta'] = 0
        
        # Convert Action values
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1], 0.0, 1.0)) 
        brake = float(np.clip(action[2], 0.0, 1.0)) 
        
        current_speed = self.S.d.get('speedX', 0)

        # Kickstart Logic - help the car start moving
        if self.time_step < 50 and current_speed < 5.0:
            accel = 1.0
            brake = 0.0

        self.R.d['steer'] = steer
        self.R.d['accel'] = accel
        self.R.d['brake'] = brake
        
        # Gear management
        if current_speed < 1.0:
            self.R.d['gear'] = 1
        self._automatic_gear_shifting()

        # Send Action to server
        self._send_to_server(self.R)
        
        # Receive State from server
        server_str = self._recv_from_server()
        
        if not server_str:
            print("Server not responding inside step. Sending Restart Signal (meta=1)...")
            self.R.d['meta'] = 1
            self._send_to_server(self.R)
            return np.zeros(29, dtype=np.float32), 0.0, True, False, {'error': 'server_disconnect'}
            
        self.S.parse_server_str(server_str)
        
        # --- LOGGING & PRINTING ---
        # Save to CSV
        log_data = self.logger.log_step(self.time_step, self.S.d, self.R)

        # Print to Console every 1000 steps
        if self.time_step % 1000 == 0: 
            print(f"ITER {log_data['Iteration']} | Spd: {log_data['Speed_X(km/h)']} | "
                  f"Steer: {log_data['Steer_Angle']} | Pos: {log_data['Track_Pos']} | "
                  f"Brake: {log_data['Brake']} | Gear: {log_data['Gear']}")

        # Calculate Observation & Reward
        obs = self._make_observation(self.S.d)
        reward, done = self._calculate_reward(self.S.d, action)
        
        # Stuck Detection - prevent car from being stuck
        if current_speed < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter > 150 and self.time_step > 50:
                reward -= 50.0  
                done = True
                print(f"STUCK DETECTED! Step: {self.time_step}")
        else:
            self.stuck_counter = 0

        # Check for shutdown/restart signals
        if '***shutdown***' in server_str or '***restart***' in server_str:
            done = True
        
        self.time_step += 1
        
        # Send Reset Signal if episode is done
        if done:
            if self.time_step > 20:
                t_pos = self.S.d.get('trackPos', 0)
                sp_x = self.S.d.get('speedX', 0)
                print(f"DIED! Step: {self.time_step} | Reason: TrackPos={t_pos:.2f}, Speed={sp_x:.2f}")
            
            self.R.d['meta'] = 1 
            self._send_to_server(self.R)

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        self.stuck_counter = 0
        self._relaunch_client()
        
        # Wait for valid state from server
        max_retries = 100
        for i in range(max_retries):
            if i > 0: time.sleep(0.02)
            server_str = self._recv_from_server()
            if server_str:
                self.S.parse_server_str(server_str)
                t_pos = self.S.d.get('trackPos', 100.0)
                if abs(t_pos) < 0.5 and i > 5:
                    break
        
        if self.S.d:
            obs = self._make_observation(self.S.d)
        else:
            obs = np.zeros(29, dtype=np.float32)
        return obs, {}

    def close(self):
        if self.so:
            self.so.close()
            self.so = None

    def _automatic_gear_shifting(self):
        """Automatic gear shifting based on speed - Sports Mode for max acceleration"""
        speed = self.S.d.get('speedX', 0)
        gear = self.R.d['gear']
        
        # Redline shifting for maximum acceleration
        if speed < 0: gear = 1
        elif speed < 65:  gear = 1  
        elif speed < 105: gear = 2  
        elif speed < 145: gear = 3  
        elif speed < 185: gear = 4  
        elif speed < 230: gear = 5  
        else: gear = 6
        
        self.R.d['gear'] = gear

    def _make_observation(self, state_dict):
        """Convert server state to normalized observation vector"""
        # Track sensors (19 values) - distance to track edges
        track = np.array(state_dict.get('track', [0]*19)) / 200.0
        
        # Other features (10 values)
        others = np.array([
            state_dict.get('speedX', 0) / 300.0,  # Forward speed
            state_dict.get('speedY', 0) / 300.0,  # Lateral speed
            state_dict.get('speedZ', 0) / 300.0,  # Vertical speed
            state_dict.get('angle', 0) / PI,      # Angle to track axis
            state_dict.get('trackPos', 0),        # Position on track (-1 to 1)
            state_dict.get('rpm', 0) / 10000.0,   # Engine RPM
            state_dict.get('wheelSpinVel', [0,0,0,0])[0] / 100.0,  # Wheel spin velocities
            state_dict.get('wheelSpinVel', [0,0,0,0])[1] / 100.0,
            state_dict.get('wheelSpinVel', [0,0,0,0])[2] / 100.0,
            state_dict.get('wheelSpinVel', [0,0,0,0])[3] / 100.0,
        ])
        return np.concatenate([track, others]).astype(np.float32)

    def _calculate_reward(self, state, action):
        """Calculate reward based on speed, position, and control smoothness"""
        sp_x = state.get('speedX', 0)
        track_pos = state.get('trackPos', 0)
        angle = state.get('angle', 0)
        brake = float(action[2])
        steer = float(action[0])
        
        # 1. Speed reward - forward progress with angle consideration
        sp_x_norm = np.clip(sp_x / 300.0, 0, 1)
        if sp_x > 0:
            # Penalize speed when angle is too large (going sideways)
            progress = (sp_x_norm ** 1.2) * 2.0 * np.cos(angle)  # 1.0 → 2.0 (increase weight)
        else:
            progress = sp_x * 0.02  # Stronger reverse penalty
        
        reward = progress
        
        # 2. Track center maintenance - stronger penalty at edges
        # Use exponential penalty for being far from center
        if abs(track_pos) < 0.5:
            reward -= (track_pos ** 2) * 0.3  # Gentle penalty near center
        else:
            reward -= (track_pos ** 2) * 1.0  # Strong penalty at edges
        
        # 3. Angle stability - penalize large angles more heavily
        reward -= (angle ** 2) * 0.5  # 0.3 → 0.5
        
        # 4. Smooth steering - prevent jerky movements
        reward -= abs(steer) * 0.05  # Reduced from 0.1
        
        # 5. Survival bonus - encourage staying on track
        reward += 0.1  # 0.05 → 0.1
        
        # 6. Low speed penalty - encourage maintaining speed
        if sp_x < 5.0:
            reward -= 1.0  # Strong penalty for very slow speeds
        elif sp_x < 20.0:
            reward -= 0.3  # Medium penalty
        
        # 7. Smart braking - only penalize unnecessary braking
        if sp_x > 50.0 and brake > 0.3:  # High speed braking is sometimes necessary
            reward -= brake * 0.1
        elif sp_x < 100.0 and sp_x > 20.0 and brake > 0.5:  # Medium speed
            reward -= brake * 0.3  # Penalize hard braking at medium speeds
        
        # 8. Reward for maintaining good racing line
        # If centered and fast, give bonus
        if abs(track_pos) < 0.3 and sp_x > 100.0 and abs(angle) < 0.1:
            reward += 0.5  # Bonus for good racing performance
        
        done = False
        
        # 9. Termination conditions
        if abs(track_pos) > 1.0:  # Track departure
            reward = -20.0  # -10 → -20 (stronger penalty)
            done = True
        elif sp_x < -5.0:  # Reverse driving
            reward = -20.0
            done = True
        elif abs(angle) > 1.5:  # Spinning out (almost perpendicular)
            reward = -15.0
            done = True
        
        return reward, done
    
    def _relaunch_client(self):
        """Relaunch connection to TORCS server"""
        if self.so: self.so.close()
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.so.settimeout(1.0)
        except:
            sys.exit(-1)

        retry_count = 0
        while True:
            init_angles = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = '%s(init %s)' % (self.sid, init_angles)
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
                data, _ = self.so.recvfrom(data_size)
                data = data.decode('utf-8')
                if '***identified***' in data:
                    print(f"Client connected on {self.port}")
                    break
            except socket.error:
                print(f"Waiting for TORCS server... ({retry_count}) - Sending Restart Signal")
                time.sleep(1)
                retry_count += 1
                if retry_count % 5 == 0:
                     dummy_action = { 'accel':0, 'brake':0, 'gear':1, 'steer':0, 'clutch':0, 'focus':0, 'meta':1 }
                     self._send_to_server_raw(dummy_action)

    def _send_to_server_raw(self, action_dict):
        """Send raw action dictionary to server"""
        if not self.so: return
        out = str()
        for k, v in action_dict.items():
            out += f"({k} {v})"
        try:
            self.so.sendto(out.encode(), (self.host, self.port))
        except socket.error:
            pass

    def _send_to_server(self, action_obj):
        """Send DriverAction object to server"""
        if not self.so: return
        try:
            self.so.sendto(repr(action_obj).encode(), (self.host, self.port))
        except socket.error:
            pass

    def _recv_from_server(self):
        """Receive state string from server"""
        if not self.so: return None
        try:
            data, _ = self.so.recvfrom(data_size)
            return data.decode('utf-8')
        except socket.error:
            return None

if __name__ == "__main__":
    env = TorcsEnv(port=3001)
    
    model_name = "torcs_sac_dirt"
    model_path = f"{model_name}.zip"

    if os.path.exists(model_path):
        print(f"▶ Found existing model: {model_path}")
        print("▶ Loading model to RESUME training...")
        model = SAC.load(model_name, env=env)
    else:
        print("▶ No existing model found.")
        print("▶ Creating NEW SAC model...")
        
        # Action noise for exploration - helps SAC explore better in continuous action space
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), 
            sigma=0.1 * np.ones(n_actions)
        )
        
        model = SAC(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=0.0003,           # Learning rate for optimizer
            buffer_size=1000000,            # Size of replay buffer
            learning_starts=1000,           # Start learning after this many steps
            batch_size=256,                 # Batch size for training
            tau=0.005,                      # Soft update coefficient for target networks
            gamma=0.99,                     # Discount factor
            train_freq=1,                   # Update policy every n steps
            gradient_steps=1,               # Number of gradient steps per update
            action_noise=action_noise,      # Action noise for exploration
            ent_coef='auto',                # Entropy coefficient (auto-tuned)
            target_update_interval=1,       # Update target network every n steps
            target_entropy='auto',          # Target entropy (auto-tuned)
        )

    print("Learning started... (Press Ctrl+C to stop and save)")
    try:
        model.learn(total_timesteps=1000000, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    model.save(model_name)
    print(f"Model saved to {model_name}.zip")