# 🏎️ TORCS SAC Agent Training Report: Problem-Solving Log

## 1. Challenge: The 2400m Wall (S-Curve)
- **Problem**: The agent consistently crashed at the **2400m S-Curve**. It optimized for high speed but failed to brake for the complex geometry.
- **Hypothesis**: The penalty for crashing (-500) was outweighed by the accumulated speed reward (+2000 from the first 2400m), making "crashing fast" a valid strategy.
- **Solution**: 
  1. Implemented **Balanced Crash Penalty** (`-200 - dist/10`) to scale punishment with distance.
  2. Implemented **Amplified Milestone Rewards** (x2 Bonus for >2000m segments).
- **Result**: 
  - The agent broke the 2400m barrier.
  - Reached a new record distance of **3,311m**.

## 2. Challenge: Agent Conservatism ("Stuck" Behavior)
- **Problem**: After increasing crash penalties, the agent became risk-averse. Instead of driving, it chose to stop (velocity = 0).
  - **Data**: **12.6%** of all steps were spent at < 5km/h.
  - **Data**: **62%** of recent episodes ended due to "Stuck" timeout rather than crashing.
- **Hypothesis**: The "Stuck" detection logic waited 150 steps (3 seconds) before punishing. The agent learned to "park" and accept a small penalty rather than risk a large crash penalty.
- **Solution**: **Immediate Termination** strategy.
  - Changed logic to immediately terminate the episode with a `-10.0` penalty if speed drops below **20 km/h**.
- **Result**:
  - **Low-speed steps dropped drastically**: **12.6% → 3.7%** 📉
  - Agent creates continuous momentum; "parking" behavior is eliminated.

## 3. Challenge: Logging Reliability
- **Problem**: A bug in the file path caused 2 hours of training data to be logged to an old file (`v1` instead of `v2`), creating a "blind spot" in monitoring.
- **Solution**: 
  - Merged missing logs back into `v2`.
  - Added a `log_event()` function to the logger to record system events (Rollbacks, Model Loads) directly into the CSV.
- **Result**: Full traceability of training history restored.

## 4. Current Milestone: Lap Completions
- **Achievement**: The agent has successfully completed the full 3600m Corkscrew lap multiple times across training sessions.
- **Completion Statistics**:
  - **Initial Training (`torcs_sac_telemetry_v2.csv`)**: **2 completions**
    - Max Distance: **3,600.48m**
    - Completion distances: 3,600.48m, 3,600.19m
  - **Extended Training (`torcs_sac_hybrid_10k.csv`)**: **933 completions** 🏆
    - Max Distance: **3,618.63m**
    - Total training steps: **9,745,365**
- **Status**: Training is highly stable, with consistent lap completions and the primary goal (Completion) achieved and exceeded significantly.
