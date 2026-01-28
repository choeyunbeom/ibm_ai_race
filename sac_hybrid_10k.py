
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
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

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

# --- Pandas Logger Class (Separate file for Hybrid Exp) ---
class PandasLogger:
    def __init__(self, filename="torcs_sac_hybrid_10k.csv"):
        self.filename = filename
        self.columns = [
            "Iteration", "Time", "Speed_X(km/h)", "Target_Speed_Est", "Distance(m)", 
            "Steer_Angle", "Steer_Gain_Est", "Track_Pos", "Centering_Gain_Est",   
            "Brake", "Brake_Threshold_Est", "Accel", "Gear", "RPM", "Wheel_Spin"          
        ]
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.filename, index=False)

    def log_step(self, iteration, state, action):
        sp_x = state.get('speedX', 0.0)
        track_pos = state.get('trackPos', 0.0)
        rpm = state.get('rpm', 0.0)
        dist = state.get('distRaced', 0.0)
        gear = action.d['gear']
        steer = action.d['steer']
        accel = action.d['accel']
        brake = action.d['brake']
        
        wheel_spin_vel = state.get('wheelSpinVel', [0,0,0,0])
        avg_wheel_spin = sum(wheel_spin_vel) / 4.0 if wheel_spin_vel else 0.0

        new_data = {
            "Iteration": iteration,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Speed_X(km/h)": round(sp_x, 2),
            "Target_Speed_Est": round(sp_x, 2),
            "Distance(m)": round(dist, 2),
            "Steer_Angle": round(steer, 4),
            "Steer_Gain_Est": round(abs(steer), 4),
            "Track_Pos": round(track_pos, 4),
            "Centering_Gain_Est": round(abs(steer) * abs(track_pos), 4),
            "Brake": round(brake, 4),
            "Brake_Threshold_Est": round(brake, 4),
            "Accel": round(accel, 4),
            "Gear": gear,
            "RPM": round(rpm, 0),
            "Wheel_Spin": round(avg_wheel_spin, 2)
        }
        pd.DataFrame([new_data]).to_csv(self.filename, mode='a', header=False, index=False)
        return new_data

    def log_event(self, message):
        """Append a comment/event line to the log file."""
        try:
            with open(self.filename, 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"# EVENT: {message} at {timestamp}\n")
        except:
            pass

# --- GYM ENVIRONMENT ---
class TorcsEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, host='localhost', port=3001, sid='SCR'):
        super(TorcsEnv, self).__init__()
        self.host = host
        self.port = port
        self.sid = sid
        
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

        self.so = None
        self.S = ServerState()
        self.R = DriverAction()
        self.time_step = 0
        self.stuck_counter = 0
        self.max_distance = 0
        
        self.logger = PandasLogger("torcs_sac_hybrid_10k.csv") # Hybrid Experiment Log

    def step(self, action):
        self.R.d['meta'] = 0
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1], 0.0, 1.0)) 
        brake = float(np.clip(action[2], 0.0, 1.0)) 
        
        current_speed = self.S.d.get('speedX', 0)
        self.R.d['steer'] = steer
        self.R.d['accel'] = accel
        self.R.d['brake'] = brake
        
        if current_speed < 1.0: self.R.d['gear'] = 1
        if current_speed < 1.0: self.R.d['gear'] = 1
        self._automatic_gear_shifting()

        # [Aggressive Launch Control]
        if self.time_step < 200 and current_speed < 30.0:
             self.R.d['accel'] = 1.0 # FULL POWER
             self.R.d['brake'] = 0.0 # NO BRAKE
             # self.R.d['steer'] = 0.0 # Allow steering to avoid wall
            #  self.R.d['gear'] = 1    # 1ST GEAR FORCE

        self._send_to_server(self.R)
        server_str = self._recv_from_server()
        
        if not server_str:
            print("Server not responding. Restarting...")
            self.R.d['meta'] = 1
            self._send_to_server(self.R)
            return np.zeros(29, dtype=np.float32), 0.0, True, False, {'error': 'server_disconnect'}
            
        self.S.parse_server_str(server_str)
        log_data = self.logger.log_step(self.time_step, self.S.d, self.R)

        if self.time_step % 100 == 0: 
            print(f"SAC STEP {self.time_step} | Spd: {log_data['Speed_X(km/h)']} | Dist: {log_data['Distance(m)']}")

        obs = self._make_observation(self.S.d)
        reward, done = self._calculate_reward(self.S.d, action)
        
        if current_speed < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter > 150 and self.time_step > 50:
                dist = self.S.d.get('distRaced', 0)
                reward = -200.0 - (dist / 10.0)  # [Anti-Game] Match relaxed crash penalty
                done = True
                print(f"STUCK DETECTED! Step: {self.time_step}")
        else:
            self.stuck_counter = 0

        if '***shutdown***' in server_str or '***restart***' in server_str:
            done = True
        
        self.time_step += 1
        
        if done:
            dist = self.S.d.get('distRaced', 0)
            if dist > 3600.0:
                 print(f"🏆 RACE FINISHED! (3600m / 1 Laps Completed) 🏆", flush=True)
            elif self.time_step > 20: 
                 print(f"DIED! Step: {self.time_step} | Speed={self.S.d.get('speedX', 0):.2f}")
            
            self.R.d['meta'] = 1 
            self._send_to_server(self.R)

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        self.stuck_counter = 0
        self.max_distance = 0
        self._relaunch_client()
        
        max_retries = 100
        for i in range(max_retries):
            if i > 0: time.sleep(0.02)
            server_str = self._recv_from_server()
            if server_str:
                self.S.parse_server_str(server_str)
                if abs(self.S.d.get('trackPos', 100.0)) < 0.5 and i > 5:
                    break
        
        if self.S.d: obs = self._make_observation(self.S.d)
        else: obs = np.zeros(29, dtype=np.float32)
        return obs, {}

    def close(self):
        if self.so:
            self.so.close()
            self.so = None

    def _automatic_gear_shifting(self):
        speed = self.S.d.get('speedX', 0)
        gear = self.R.d['gear']
        # [Revert] Back to Standard Safe Tunning
        if speed < 0: gear = 1
        elif speed < 50:  gear = 1   # Back to 50 (Safe)
        elif speed < 90:  gear = 2   # Back to 90
        elif speed < 130: gear = 3   # Back to 130
        elif speed < 170: gear = 4   # Back to 170
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
        # [V2] Speed-Optimized Reward Function (Simplified)
        sp_x = state.get('speedX', 0)
        track_pos = state.get('trackPos', 0)
        angle = state.get('angle', 0)
        dist = state.get('distRaced', 0)
        brake = float(action[2])

        # 1. Speed Reward (x3.0 Multiplier)
        sp_x_norm = np.clip(sp_x / 300.0, 0, 1)
        if sp_x > 0:
            reward = (sp_x_norm ** 2.0) * 4.0 * np.cos(angle)  # x4.0 boost
        else:
            reward = sp_x * 0.01  # Reverse penalty

        # 2. Track Center Reward
        if sp_x > 30.0:
            deviation = max(0, abs(track_pos) - 0.25) 
            reward += (1.0 - deviation) * 2.0  # Increased center reward
            
        # 3. Angle Stability & Steering
        reward -= (angle ** 2) * 0.5  # Moderate angle penalty
        reward -= abs(action[0]) * 0.05  # Light steering penalty
        
        # 4. Progressive Low Speed Penalty
        if self.time_step > 100:  # [Warmup] No penalty for first 100 steps
            if sp_x < 20.0:
                reward = -10.0 # [Anti-Stuck] Immediate Termination for crawling
                done = True 
                print(f"TOO SLOW! Step: {self.time_step} | Speed={sp_x:.2f}")
                return reward, done
            elif sp_x < 50.0:
                reward -= 1.0  # [Relaxed] Moderate penalty for very slow
            elif sp_x < 70.0:
                reward -= 1.0  # Moderate penalty
        
        # 4.5 Speed Limit for 2400m S-Curve (Hardcoded)
        if 2400.0 < dist < 2500.0 and sp_x > 100.0:
            reward -= (sp_x - 100.0) * 0.3  # Penalty for overspeeding in S-curve
        
        # 5. Straight Line Speed Bonus
        track_sensors = state.get('track', [0]*19)
        front_dist = track_sensors[9]
        if front_dist > 100.0:
            if sp_x > 150.0:
                reward += 1.0
            elif sp_x > 120.0:
                reward += 0.5
            elif sp_x > 100.0:
                reward += 0.2

        # 6. Milestone Rewards (Simplified)
        # Target: 3650.0m (3600m + 50m buffer) ensures game engine registers finish.
        RACE_LENGTH = 3608.0 
        
        if dist >= RACE_LENGTH:
            reward = 1000.0  # Race Finish Bonus
            done = True
            return reward, done

        if dist > self.max_distance:
            start_m = int(self.max_distance / 100) + 1
            end_m = int(dist / 100)
            for m_idx in range(start_m, end_m + 1):
                total_m = m_idx * 100
                local_m = total_m % 3600
                if local_m == 0:
                    continue
                # [Amplify Milestone Bonus]
                base_bonus = local_m / 100.0  # Base logic
                if local_m > 3000:
                    bonus = base_bonus * 3.0  # x3 Super Bonus for >3000m
                elif local_m > 2000:
                    bonus = base_bonus * 2.0  # x2 Bonus for >2000m
                else:
                    bonus = base_bonus
                reward += bonus
                print(f"MILESTONE {local_m:.0f}m REACHED! (+{bonus:.2f})", flush=True)
            self.max_distance = dist

        # 7. Termination
        done = False
        if abs(track_pos) > 1.2: # [Relaxed] 1.0 -> 1.2 to allow curb riding (Essential for S-Curve)
            # [Distance-Proportional Penalty - Moderate Rollback]
            # -200 at 0m, -440 at 2400m (Balance between safety and exploration)
            reward = -200.0 - (dist / 10.0)
            if self.time_step < 100:
                reward -= 20.0
            done = True
        elif sp_x < -5.0:
            reward = -10.0
            done = True
        
        return reward, done
    
    # --- Networking Methods ---
    def _relaunch_client(self):
        if self.so: self.so.close()
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.so.settimeout(1.0)
        except: sys.exit(-1)
        retry_count = 0
        while True:
            init_angles = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = '%s(init %s)' % (self.sid, init_angles)
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
                data, _ = self.so.recvfrom(data_size)
                if '***identified***' in data.decode('utf-8'):
                    print(f"Client connected on {self.port}")
                    break
            except socket.error:
                time.sleep(1)
                retry_count += 1
                if retry_count % 5 == 0:
                     dummy = { 'accel':0, 'brake':0, 'gear':1, 'steer':0, 'clutch':0, 'focus':0, 'meta':1 }
                     self._send_to_server_raw(dummy)

    def _send_to_server_raw(self, d):
        if not self.so: return
        out = ''.join([f"({k} {v})" for k,v in d.items()])
        try: self.so.sendto(out.encode(), (self.host, self.port))
        except: pass

    def _send_to_server(self, obj):
        if not self.so: return
        try: self.so.sendto(repr(obj).encode(), (self.host, self.port))
        except: pass

    def _recv_from_server(self):
        if not self.so: return None
        try:
            data, _ = self.so.recvfrom(data_size)
            return data.decode('utf-8')
        except: return None

if __name__ == "__main__":
    print("▶ CLEAN RESUME: Loading Pre-Streak Golden Model")
    env = TorcsEnv(port=3001)
    
    # [Golden Model]
    model_path = "checkpoints_sac_hybrid/torcs_sac_corkscrew_hybrid_3247596_steps"
    print(f"Loading: {model_path}")
    
    # Load
    model = SAC.load(model_path, env=env)
    
    # Replay Buffer
    rb_path = model_path + "_replay_buffer.pkl"
    if os.path.exists(rb_path):
        print("Buffer Loaded.")
        model.load_replay_buffer(rb_path)
        
    # [Proper Noise - 0.1 for flexibility]
    n_actions = env.action_space.shape[0]
    model.action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    print("RESUMING TRAINING (Clean State)...")
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints_clean/', name_prefix='torcs_sac_clean')
    # [Safety] Save on Ctrl+C (Restored)
    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted! Saving progress...")
    finally:
        save_path = "checkpoints_clean/torcs_sac_clean_final"
        model.save(save_path)
        model.save_replay_buffer(save_path + "_replay_buffer")
        print(f"✅ Model & Buffer Saved to {save_path}")
        env.close()
