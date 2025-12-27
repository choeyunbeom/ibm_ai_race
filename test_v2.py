import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import sys
import time
import math
import os
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

    def step(self, action):
        # 0. Set default meta value (0: Racing)
        self.R.d['meta'] = 0
        
        # 1. Convert Action values (PPO Output -> TORCS Input)
        # Steer: -1.0 ~ 1.0
        steer = float(np.clip(action[0], -1.0, 1.0))
        # Accel, Brake: 0.0 ~ 1.0 (Clip negative PPO outputs to 0)
        accel = float(np.clip(action[1], 0.0, 1.0)) 
        brake = float(np.clip(action[2], 0.0, 1.0)) 
        
        # Get current speed (based on previous frame)
        current_speed = self.S.d.get('speedX', 0)

        # -----------------------------------------------------------
        # [Kickstart] Early forced start logic
        # -----------------------------------------------------------
        # If speed is under 5km/h for the first 1 sec (50 steps) of the game,
        # ignore AI decision and force full throttle to start.
        # (Prevents AI from getting scared/stuck at the starting line)
        if self.time_step < 50 and current_speed < 5.0:
            accel = 1.0
            brake = 0.0

        # Apply converted values
        self.R.d['steer'] = steer
        self.R.d['accel'] = accel
        self.R.d['brake'] = brake
        
        # Gear control (Keep 1st gear when stopped/low speed)
        if current_speed < 1.0:
            self.R.d['gear'] = 1
        self._automatic_gear_shifting()

        # 2. Send Action to server
        self._send_to_server(self.R)
        
        # 3. Receive State from server
        server_str = self._recv_from_server()
        
        # -----------------------------------------------------------
        # [Soft Restart] Handle connection loss
        # -----------------------------------------------------------
        # If no server response, do not close the program,
        # treat this turn as Done, but send 'meta=1' to trigger restart.
        if not server_str:
            print("Server not responding inside step. Sending Restart Signal (meta=1)...")
            self.R.d['meta'] = 1
            self._send_to_server(self.R)
            return np.zeros(29, dtype=np.float32), 0.0, True, False, {'error': 'server_disconnect'}
            
        # Parse data
        self.S.parse_server_str(server_str)
        
        # 4. Calculate Observation and Reward
        obs = self._make_observation(self.S.d)
        reward, done = self._calculate_reward(self.S.d, action)
        
        # -----------------------------------------------------------
        # [Stuck Detection]
        # -----------------------------------------------------------
        # If Kickstart period (50 steps) has passed
        # and speed remains under 1km/h for over 3 seconds (150 steps), force reset.
        if current_speed < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter > 150 and self.time_step > 50:
                reward -= 50.0  # Strong penalty for wasting time
                done = True
                print(f"STUCK DETECTED! Step: {self.time_step}")
        else:
            self.stuck_counter = 0 # Reset counter if moving

        # Handle server shutdown signals
        if '***shutdown***' in server_str or '***restart***' in server_str:
            done = True
        
        self.time_step += 1
        
        # -----------------------------------------------------------
        # [Reset Signal] Handle episode termination
        # -----------------------------------------------------------
        # Command server to "Restart (meta=1)" when dead or finished.
        if done:
            # Log output (Excluding very short episodes)
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
        if speed < 0: gear = 1
        elif speed < 50: gear = 1
        elif speed < 80: gear = 2
        elif speed < 120: gear = 3
        elif speed < 150: gear = 4
        elif speed < 190: gear = 5
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

        # -----------------------------------------------------------
        # [Mod 1] Curb reward inflation (* 10.0 -> * 5.0)
        # -----------------------------------------------------------
        sp_x_norm = sp_x / 300.0
        if sp_x > 0:
            # Previous 10.0 was too high, making AI unafraid of death.
            # Reduced to 5.0 to slow down "earning rate".
            progress = (sp_x_norm ** 1.5) * 5.0 * np.cos(angle)
        else:
            progress = 0.0 
        
        reward = progress

        # 2. Minor penalties (Maintained)
        reward -= abs(track_pos) * 0.05
        reward -= abs(angle) * 0.05
        reward -= abs(action[0]) * 0.05

        # 3. Survival reward (Maintained)
        reward += 0.01 

        # 4. Stagnation and low-speed penalty
        if sp_x < 5.0:
            reward -= 0.1

        # 5. Prevent unnecessary braking
        if sp_x < 80.0:
             reward -= brake * 0.05

        done = False

        # -----------------------------------------------------------
        # [Mod 2] Significantly increase death penalty (-50 -> -200)
        # -----------------------------------------------------------
        if abs(track_pos) > 1.0 or sp_x < -5.0:
            # Now losing 200 points upon death. 
            # (Points earned from ~40-50s of full throttle are lost at once)
            reward = -200.0 
            
            # Punish early death (trolling) even harder
            if self.time_step < 300:
                reward -= 100.0 # Total -300.0
            
            done = True
            
        return reward, done
    
    def _relaunch_client(self):
        if self.so: self.so.close()
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.so.settimeout(1.0) # 1 second timeout
        except:
            sys.exit(-1)

        retry_count = 0
        while True:
            # 1. Prepare initialization (restart) request message
            init_angles = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = '%s(init %s)' % (self.sid, init_angles)
            
            try:
                # 2. Send restart request to server
                self.so.sendto(initmsg.encode(), (self.host, self.port))
                
                # 3. Wait for response
                data, _ = self.so.recvfrom(data_size)
                data = data.decode('utf-8')
                
                if '***identified***' in data:
                    print(f"Client connected on {self.port}")
                    break
                    
            except socket.error:
                # If no response, log and retry (Keep program running)
                # TORCS restarts the race upon receiving the (init) message again.
                print(f"Waiting for TORCS server... ({retry_count}) - Sending Restart Signal")
                time.sleep(1)
                retry_count += 1
                
                # [Added] If no response for too long (5s), try sending a meta=1 packet.
                if retry_count % 5 == 0:
                     # Send dummy packet meaning "Force Restart!"
                     dummy_action = { 'accel':0, 'brake':0, 'gear':1, 'steer':0, 'clutch':0, 'focus':0, 'meta':1 }
                     self._send_to_server_raw(dummy_action)
    def _send_to_server_raw(self, action_dict):
        if not self.so: return
        
        # Simulate DriverAction class format to create string
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

# --- Execution (Including Resume Logic) ---
if __name__ == "__main__":
    env = TorcsEnv(port=3001)
    
    model_name = "torcs_ppo_result"
    model_path = f"{model_name}.zip"

    # 1. Check if model file exists
    if os.path.exists(model_path):
        print(f"▶ Found existing model: {model_path}")
        print("▶ Loading model to RESUME training...")
        # [Important] Load existing model (env connection required)
        model = PPO.load(model_name, env=env)
    else:
        print("▶ No existing model found.")
        print("▶ Creating NEW model...")
        # Create new model if none exists
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
            clip_range=0.2,
            ent_coef=0.01
        )

    print("Learning started... (Press Ctrl+C to stop and save)")
    try:
        # [Important] reset_num_timesteps=False: Continue logging step counts
        model.learn(total_timesteps=1000000, reset_num_timesteps=False)
    except KeyboardInterrupt:
        pass
    
    # Save on exit
    model.save(model_name)
    print(f"Model saved to {model_name}")