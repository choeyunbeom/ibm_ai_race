# 🏎️ TORCS Corkscrew Challenge: Reinforcement Learning Journey

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TORCS](https://img.shields.io/badge/TORCS-1.3.7-green.svg)](http://torcs.sourceforge.net/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **From 0% to 37 Completions: A Data-Driven Approach to Solving Autonomous Racing Challenges**

This repository documents our journey training reinforcement learning agents to autonomously complete the TORCS Corkscrew track. Rather than just presenting final results, we focus on the **problem-solving process**: identifying failures, analyzing data, and implementing systematic solutions.

##  Project Overview

**Challenge**: Train an RL agent to complete a 3,600m racing track with complex geometry, balancing speed and safety.

**Achievements**:
- Broke the 2400m barrier (100% → 35% failure rate)
- Eliminated "parking" behavior (12.6% → 3.7% low-speed steps)
- Achieved **37 track completions** over 4,349 episodes
- Best lap time: **1:48**
- Systematic debugging methodology documented

**Key Insight**: The 52.59% early failure rate isn't a bug—it's evidence of healthy exploration preventing premature convergence to suboptimal policies.

## Training Statistics

```
Algorithm: Soft Actor-Critic (SAC)
Total Steps: 9,745,365
Total Episodes: 4,349
Completion Rate: 0.85% (37 completions)
Average Distance: 1,360m
Max Distance: 3,618m
```

### Distance Distribution

| Range | Episodes | % | Interpretation |
|-------|----------|---|----------------|
| 0-1000m | 2,287 | 52.59% | Exploration phase |
| 1000-2000m | 509 | 11.70% | Mid-track learning |
| 2000-3000m | 1,406 | 32.33% | **S-Curve bottleneck** |
| 3000-3600m | 110 | 2.53% | Near completion |
| ≥3600m | 37 | 0.85% | Success |

**Finding**: Only 3.38% reached >3000m, indicating the final section is **15x harder** than reaching the midpoint.

## Repository Structure

```
torcs-rl-project/
├── README.md                    # This file
│
│
├── docs/                        # Detailed documentation
│   ├── progress_report.md       # SAC training chronicle
│   ├── progress_report_ppo.md   # PPO training attempts
│   ├── troubleshooting_report.md  # Debugging log
│   ├── sac_analysis.pdf # Interactive analysis
│   └── ppo_analysis.pdf        # PPO failure analysis
│
├── sac_clean_resume.py
├── sac_hybrid_10k.py
├── gym_torcs.py
├── snakeoil3_gym.py
├── autostart.sh
├── practice.xml
├── requirements.txt
├── example_experiment.py
├── vtorcs-RL-color/
│
│
└── assets/                      # Visualizations
    ├── sac_cumulative_progress.png
    ├── sac_distance_distribution.png
    └── sac_success_rate.png
```

## Key Problems Solved

### Problem #1: The 2400m Wall
**Symptom**: 100% crash rate at S-Curve  
**Root Cause**: Reward imbalance made "crashing fast" more rewarding than "driving safe"  
**Solution**: Scaled crash penalty based on distance traveled  
**Result**: Breakthrough to 3,311m

### Problem #2: Parking Behavior
**Symptom**: 62% of episodes ended by "stuck" timeout  
**Data**: 12.6% of all steps were at <5 km/h  
**Root Cause**: Agent learned to "park" and accept small penalty vs large crash penalty  
**Solution**: Immediate termination at <20 km/h  
**Result**: Low-speed steps dropped to 3.7%

### Problem #3: PPO Catastrophic Forgetting
**Symptom**: Performance regressed from 1,400m → 400m during fine-tuning  
**Root Cause**: Aggressive learning rate (0.0001) + imbalanced rewards (2.5x speed)  
**Solution**: Conservative reset with LR 0.00005  
**Lesson**: "More training ≠ Better performance"

## Documentation

### Main Article
**[Full Technical Blog Post](https://choeyunbeom.github.io/reinforcement%20learning/autonomous%20driving/torcs-rl-journey/)** - Complete story with methodology and insights

### Detailed Reports
- [SAC Progress Report](docs/progress_report.md) - Problem-solving chronicle
- [Troubleshooting Log](docs/sac_report_en.md) - Detailed debugging process
- [PPO Analysis](docs/progress_report_ppo.md) - Catastrophic forgetting case study
- [Interactive Analysis](docs/troubleshooting_analysis.pdf) - Visual debugging tools

## Key Results

### SAC Learning Progression
```
Episode Range    Max Distance    Key Event
0-500           1,200m          Basic control learning
500-1000        2,400m          Reached S-Curve barrier
1000-2000       3,311m          Broke through S-Curve
2000-3000       3,600m          First completion
3000+           3,618m          37 completions achieved
```

### Algorithm Comparison

| Metric | SAC | PPO |
|--------|-----|-----|
| Sample Efficiency | High | Lower |
| Training Stability | Sensitive | Robust |
| Best Distance | 3,618m | 1,400m |
| Completions | 37 | 0 |

## Lessons Learned

### 1. Reward Engineering is Critical
Small reward changes cause massive behavioral shifts. Every component must be tested for unintended exploits.

### 2. Data-Driven Debugging is Essential
Our intuition about failures was often wrong. The 12.6% low-speed metric revealed the true problem.

### 3. Metrics Can Be Deceptive
0.85% completion rate looks poor, but 52.59% early failures indicate healthy exploration, not failure.

### 4. More Training ≠ Better Performance
PPO's 65% regression (1,400m → 400m) proved that training duration must be carefully managed.

### 5. Algorithm Selection Matters
SAC's off-policy learning was crucial for sample efficiency in this sparse-reward, long-episode task.

## Technical Highlights

### Reward Function Evolution

```python
# Version 1: Naive (Failed - encouraged "crash fast")
reward = distance * 0.1 - 500 * crashed

# Version 2: Balanced (Partial - encouraged "parking")
reward = distance * 0.1 - (200 + distance/10) * crashed

# Version 3: Momentum-enforcing (Success - 37 completions)
reward = distance * 0.1 + speed_bonus + center_bonus + survival_bonus
if speed < 20: terminate_immediately()
```

### Data Analysis Methodology

```python
# Discovery of parking behavior
df['low_speed_pct'] = df['low_speed_steps'] / df['steps'] * 100
print(f"Low-speed: {df['low_speed_pct'].mean():.1f}%")  # Output: 12.6%

# Termination analysis
print(df['termination'].value_counts())
# Stuck: 62%, Crash: 38% → Led to immediate termination solution
```

## Future Work

### Short-Term
- [ ] Improve completion rate: 0.85% → >50%
- [ ] Optimize lap time: 1:48 → <1:30
- [ ] Implement curriculum learning

### Long-Term
- [ ] Multi-track generalization
- [ ] Hierarchical RL (strategy + control)
- [ ] Sim-to-real transfer

## References

1. Haarnoja, T., et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
2. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"
3. [TORCS - The Open Racing Car Simulator](http://torcs.sourceforge.net/)
4. [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

## Contributing

This project was developed for an IBM competition, but we welcome discussions and suggestions:

- Open an issue for questions
- Report bugs or unexpected behaviors
- Suggest improvements to methodology

## Contact

- **Author**: Yunbeom Choe, Zhiheng Wang, Vishal Saravanan, Saif ur Rehman 
- **Program**: MSc Data Science and AI, University of Liverpool (2025-2026)  
- **Email**: 
  - sgychoe@liverpool.ac.uk
  - z.wang252@liverpool.ac.uk
  - v.saravanan@liverpool.ac.uk 
  - sgsrehm1@liverpool.ac.uk
  - 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **IBM** for organizing the competition
- **TORCS Community** for the simulation environment
- **Stable-Baselines3 Team** for excellent RL implementations
- **University of Liverpool** for academic support

---

