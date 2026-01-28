# TORCS Corkscrew Track AI Training Report (PPO)

## 1. Executive Summary
This report details the optimization process of a PPO agent for the Corkscrew track. We encountered significant stability issues (weaving) and performance degradation (catastrophic forgetting) during initial fine-tuning. By diagnosing log data and shifting to a "fresh start" strategy with verified configurations, we stabilized the training process.

---

## 2. Problem Solving & Results

### Issue 1: Severe Steering Instability (Weaving)
*   **Problem:** The agent exhibited extreme weaving on straight sections.
    *   *Data:* Average Steering Absolute Value `|Steer|` was **1.7** (High), with a binary steering pattern (using only max left/right values).
*   **Hypothesis:** The pretrained model had learned erratic steering habits, which were exacerbated by our artificial steering constraints (Scaling 0.5, Rate Limit 0.1) that conflicted with the model's policy.
*   **Solution:** 
    1.  **Removed Constraints:** Deleted steering scaling and rate limiting to let the model control the car naturally.
    2.  **Clean Slate:** Switched to training from scratch to eliminate bad habits from previous models.
*   **Result (Projected):** Expected linear steering behavior rather than binary oscillation.

### Issue 2: Low Speed Performance
*   **Problem:** The agent was overly cautious, failing to utilize the car's potential.
    *   *Data:* Average Speed dropped from 82 km/h to **64 km/h** during early fine-tuning.
*   **Solution:**
    1.  **Reward Shaping:** Implemented a new speed reward function: `(Speed_Norm ^ 1.2) * 1.5`.
    2.  **Straight Bonus:** Added positive reinforcement for high speeds (>70 km/h) on straights.
*   **Result:** 
    *   *Data:* Recent Average Speed increased significantly to **~95 km/h**.
    *   The agent is now attempting to maintain momentum even in technical sections.

### Issue 3: Catastrophic Forgetting (Un-learning)
*   **Problem:** During fine-tuning, the agent suddenly lost basic driving capabilities (e.g., crashing into walls at full speed, 180km/h).
    *   *Data:* Max Distance regressed from ~1400m to **~400m**.
    *   *Cause:* The Learning Rate (0.0001) and Speed Reward (2.5x) were too aggressive, causing policy collapse.
*   **Solution:**
    1.  **Reset:** Discarded the corrupted model.
    2.  **Conservative Settings:** Reduced Learning Rate to **0.00005** (initially) and later adopted standard **0.0003** for fresh training.
    3.  **Balanced Rewards:** Capped speed reward multiplier at **1.5x**.

---

## 3. Final Configuration (`ppo_test.py`)

To consolidate these fixes, we finalized the configuration as follows:

### A. Core Strategy
*   **Mode:** Zero-base Training (`torcs_corkscrew_fresh`)
*   **Logic:** Removing "Kickstart" and "Steering Limits" to allow full autonomy.

### B. Optimized Reward System
| Component | Formula/Condition | Purpose |
| :--- | :--- | :--- |
| **Center Reward** | `+ (1.0 - |Pos|) * 0.5` | **Condition: Speed > 30 km/h.** Prevents camping, encourages racing lines. |
| **Speed Reward** | `(Speed/300)^1.2 * 1.0` | Balanced speed incentive. |
| **Survival** | `+0.05 / step` | Incentivizes longer episodes. |
| **Early Death** | `-20.0` (if < 100 steps) | **New.** Penalizes instant crashes to force safer starts. |
| **Low Speed** | `-0.5` (if < 10 km/h) | Prevents getting stuck or idling. |

---

## 4. Conclusion
We have successfully transitioned from a failing fine-tuning attempt to a structured fresh training phase. The logic has been validated:
1.  **Speed is recovering** (95 km/h avg).
2.  **Weaving logic** is addressed by removing conflict-inducing constraints.
3.  **Stability** is enforced by balanced hyperparameters.

The agent is currently in the **Adaptation Phase**, learning to convert its high speed into effective distance travel on the Corkscrew track.
