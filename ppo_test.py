import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import sys
import time
import math
import os
import pandas as pd  # [Changed] Use Pandas
from datetime import datetime
from stable_baselines3 import PPO

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
            "Iteration", "Time",           # [Added] Iteration
            "Speed_X(km/h)", "Target_Speed_Est", "Distance(m)", 
            "Steer_Angle", "Steer_Gain_Est",     
            "Track_Pos", "Centering_Gain_Est",   
            "Brake", "Brake_Threshold_Est",      
            "Accel", "Gear", "RPM", "Wheel_Spin"          
        ]
        
        # Check if file exists to determine if we need to write headers
        if not os.path.exists(self.filename):
            # Create an empty DataFrame with columns and save it to initialize the file
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.filename, index=False)

    def log_step(self, iteration, state, action):
        # Extract Data
        sp_x = state.get('speedX', 0.0)
        track_pos = state.get('trackPos', 0.0)
        rpm = state.get('rpm', 0.0)
        dist = state.get('distRaced', 0.0)
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
            "Iteration": iteration,                 # [Added]
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Speed_X(km/h)": round(sp_x, 2),
            "Target_Speed_Est": round(target_speed, 2),
            "Distance(m)": round(dist, 2),
            "Steer_Angle": round(steer, 4),
            "Steer_Gain_Est": round(steer_gain, 4),
            "Track_Pos": round(track_pos, 4),
            "Centering_Gain_Est": round(centering_gain, 4),
            "Brake": round(brake, 4),
            "Brake_Threshold_Est": round(brake_threshold, 4),
            "Accel": round(accel, 4),               # [Added] Accel
            "Gear": gear,
            "RPM": round(rpm, 0),
            "Wheel_Spin": round(avg_wheel_spin, 2)
        }

        # Convert to DataFrame (single row)
        df_row = pd.DataFrame([new_data])

        # Append to CSV using Pandas
        # mode='a': Append mode
        # header=False: Do not write headers again
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
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )

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
        self.logger = PandasLogger("torcs_corkscrew_base_result.csv")

    def step(self, action):
        # 0. Set default meta value
        self.R.d['meta'] = 0
        
        # 1. Convert Action values
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1], 0.0, 1.0)) 
        brake = float(np.clip(action[2], 0.0, 1.0)) 
        
        current_speed = self.S.d.get('speedX', 0)



        self.R.d['steer'] = steer
        self.R.d['accel'] = accel
        self.R.d['brake'] = brake
        
        if current_speed < 1.0:
            self.R.d['gear'] = 1
        
        self._automatic_gear_shifting()

        # 2. Send Action
        self._send_to_server(self.R)
        
        # 3. Receive State
        server_str = self._recv_from_server()
        
        if not server_str:
            print("Server not responding inside step. Sending Restart Signal (meta=1)...")
            self.R.d['meta'] = 1
            self._send_to_server(self.R)
            return np.zeros(29, dtype=np.float32), 0.0, True, False, {'error': 'server_disconnect'}
            
        self.S.parse_server_str(server_str)
        
        # --- LOGGING & PRINTING ---
        # 1. Always Save to CSV (using Pandas)
        log_data = self.logger.log_step(self.time_step, self.S.d, self.R)

        # 2. Print to Console every 1000 steps [Changed]
        if self.time_step % 100 == 0: 
            print(f"STEP {log_data['Iteration']} | Spd: {log_data['Speed_X(km/h)']} | "
                  f"Dist: {log_data['Distance(m)']}")

        # 4. Calculate Obs & Reward
        obs = self._make_observation(self.S.d)
        reward, done = self._calculate_reward(self.S.d, action)
        
        # [Stuck Detection]
        if current_speed < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter > 150 and self.time_step > 50:
                reward -= 50.0  
                done = True
                print(f"STUCK DETECTED! Step: {self.time_step}")
        else:
            self.stuck_counter = 0

        if '***shutdown***' in server_str or '***restart***' in server_str:
            done = True
        
        self.time_step += 1
        
        # [Reset Signal]
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
        self.time_step = 0
        self.stuck_counter = 0
        self.max_distance = 0
        self.max_distance = 0
        self._relaunch_client()
        
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
        speed = self.S.d.get('speedX', 0)
        gear = self.R.d['gear']
        
        # [Optimized] Smoother Shifting
        if speed < 0: gear = 1
        elif speed < 50:  gear = 1   # Lowered from 65
        elif speed < 90:  gear = 2   # Lowered from 105
        elif speed < 130: gear = 3   # Lowered from 145
        elif speed < 170: gear = 4   # Lowered from 185
        elif speed < 220: gear = 5
        else: gear = 6
        
        self.R.d['gear'] = gear

    def _make_observation(self, state_dict):
        track = np.array(state_dict.get('track', [0]*19)) / 200.0
        others = np.array([
            state_dict.get('speedX', 0) / 300.0,
            state_dict.get('speedY', 0) / 300.0,
            state_dict.get('speedZ', 0) / 300.0,
            state_dict.get('angle', 0) / PI,
            state_dict.get('trackPos', 0),
            state_dict.get('rpm', 0) / 10000.0,
            state_dict.get('wheelSpinVel', [0,0,0,0])[0] / 100.0,
            state_dict.get('wheelSpinVel', [0,0,0,0])[1] / 100.0,
            state_dict.get('wheelSpinVel', [0,0,0,0])[2] / 100.0,
            state_dict.get('wheelSpinVel', [0,0,0,0])[3] / 100.0,
        ])
        return np.concatenate([track, others]).astype(np.float32)

    def _calculate_reward(self, state, action):
        sp_x = state.get('speedX', 0)
        track_pos = state.get('trackPos', 0)
        angle = state.get('angle', 0)
        brake = float(action[2])

        
        # 1. Speed Reward (Scaled Down)
        sp_x_norm = np.clip(sp_x / 300.0, 0, 1)

        if sp_x > 0:
            progress = (sp_x_norm ** 1.0) * 1.0 * np.cos(angle)
        else:
            progress = sp_x * 0.01  # Reverse penalty
        
        reward = progress

        dist = state.get('distRaced', 0)
        # [New] Milestone Reward
        if dist > self.max_distance:
            # Check for 100m milestones
            for m in range(100, 4000, 100):
                if self.max_distance < m and dist >= m:
                    bonus = 0
                    if m < 1000:
                        bonus = m / 100
                    elif m < 1500:
                        bonus = m / 75
                    elif m < 2000:
                        bonus = m / 50
                    elif m < 2500:
                        bonus = m / 25
                    elif m < 3000:
                        bonus = m / 20
                    elif m < 3500:
                        bonus = m / 15
                    elif m < 4000:
                        bonus = m / 10
                    
                    reward += bonus
                    print(f"TARGET MILESTONE {m}m REACHED! (+{bonus:.2f})", flush=True)
            self.max_distance = dist
        
        # 2. Track Center Reward (Tolerance Zone for Weaving Prevention)
        if sp_x > 30.0:
            # [Improved] Allow being slightly off-center (±0.15) without penalty
            # This prevents the agent from frantically steering to find perfect 0.0
            deviation = max(0, abs(track_pos) - 0.25) 
            reward += (1.0 - deviation) * 0.5
            
        # [New] Straight Line Stability (Context Aware)
        track_sensors = state.get('track', [0]*19)
        front_dist = track_sensors[9]
        
        # [Modified] Relaxed constraint to prevent oscillation
        if front_dist > 100.0:
            # Only penalize SIGNIFICANT steering on straights to allow micro-corrections
            if abs(action[0]) > 0.25: # Relaxed threshold 0.1 -> 0.25
                reward -= abs(action[0]) * 0.2 # Reduced penalty 0.5 -> 0.2

        
        # 3. Angle Stability
        reward -= (angle ** 2) * 0.3
        
        # 4. Smooth Steering (Prevent Excessive Steering)
        reward -= abs(action[0]) * 0.05 # Reduced 0.1 -> 0.05
        
        # # 5. Survival Bonus
        # reward += 0.05
        
        # 6. Low Speed Penalty (Reinforced)
        if sp_x < 10.0:
            reward -= 0.5
        
        # 7. Inappropriate Braking (Conditional)
        if sp_x < 50.0 and brake > 0.0:
            reward -= brake * 0.5  # Penalize braking when slow
        done = False
        
        # 8. Termination Conditions
        if abs(track_pos) > 1.0:  # Off track
            reward = -40.0
            
            # [Early Death Penalty]
            if self.time_step < 100:
                reward -= 20.0
            
            done = True
        elif sp_x < -5.0:  # Reverse
            reward = -10.0
            done = True
        
        return reward, done
    
    def _relaunch_client(self):
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
        if not self.so: return
        out = str()
        for k, v in action_dict.items():
            out += f"({k} {v})"
        try:
            self.so.sendto(out.encode(), (self.host, self.port))
        except socket.error:
            pass

    def _send_to_server(self, action_obj):
        if not self.so: return
        try:
            self.so.sendto(repr(action_obj).encode(), (self.host, self.port))
        except socket.error:
            pass

    def _recv_from_server(self):
        if not self.so: return None
        try:
            data, _ = self.so.recvfrom(data_size)
            return data.decode('utf-8')
        except socket.error:
            return None

if __name__ == "__main__":
    env = TorcsEnv(port=3001)
    
    new_name = "torcs_corkscrew_fresh_2" 
    model_name = "torcs_corkscrew_fresh_2"
    # [RESTORE] Loading latest saved model
    # model_path = "./checkpoints/torcs_corkscrew_fresh_2133110_steps.zip" 

    if os.path.exists(model_name + ".zip"):
        print(f"▶ Found existing model: {model_name}")
        print("▶ Loading model to RESUME training...")
        model = PPO.load(model_name, env=env)
    else:
        print("▶ No existing model found.")
        print("▶ Creating NEW model...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,  # Standard for fresh training
            ent_coef=0.01
        )

    # [Added] Autosave every 50,000 steps (approx 1-2 hours)
    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path='./checkpoints/',
        name_prefix=new_name
    )

    print("Learning started... (Press Ctrl+C to stop and save)")
    try:
        model.learn(total_timesteps=1000000, reset_num_timesteps=False, callback=checkpoint_callback)
    except KeyboardInterrupt:
        pass
    
    model.save(new_name)
    print(f"Model saved to {new_name}")