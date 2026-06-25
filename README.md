
# LTC-AN: Lightweight Temporal Convolutional Attention Network

Engineered deep learning model for real-time anomaly detection in satellite embedded telemetry systems, deployed on ARM Cortex-M / Raspberry Pi 4 edge hardware.

**Researcher:** Philip Opoku Brako | KNUST Computer Science | Student ID: 20916855  
**Supervisor:** [Insert Supervisor Name]  
**Institution:** Kwame Nkrumah University of Science and Technology (KNUST)

---

## What This Project Does
Detects faults (single-event upsets, thermal runaways, bus degradation) in real-time multivariate satellite telemetry using a fabricated TCN architecture with depthwise separable causal attention — running directly on edge hardware without ground-station reliance.

## Hardware Setup
- Arduino Mega 2560 (sensor array)
- Raspberry Pi 4B 4GB (edge flight computer simulator)
- Sensors: ACS712 voltage, DS18B20 thermal x3, MPU-6050 IMU, HMC5883L magnetometer

## Installation
```bash
git clone https://github.com/3406822/LTC-AN-Telemetry
cd LTC-AN-Telemetry
pip install -r requirements.txt
```

## Project Structure
```
LTC-AN-Telemetry/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_dataset_generation.ipynb
│   ├── 02_baseline_training.ipynb
│   └── 03_ltcan_training_evaluation.ipynb
├── src/
│   └── ltcan_model.py
└── data/
    └── sample/
```

## Citation
If you use this work, please cite:
> Brako, P. O. (2026). An Engineered Lightweight Temporal Convolutional Attention Network (LTC-AN) for Real-Time Edge-Based Anomaly Detection in Low-Earth Orbit Satellite Embedded Telemetry Systems. KNUST CS Research Proposal.

## License
MIT License
```

