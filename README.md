# Shielding-Federated-Learning
Mitigating Label Transferability against Poisoning Attacks in Cloud-edge-client System
## The code is partially available

### Creat a experimental environment: 
PYCHARM environment based on PYTORCH framework

#### Attack Settings
To create an attack environment, you can first determine the basic attack setup parameters, for example, the number of attackers, the number of all clients, and so on, which can be controlled artificially.

#### Experiment Setup
1) Surrogate models: ResNet50, DPN26, SENet18, ResNet18 and MobileNetV2
2) Dropout probability: 0.3
3) Number of epochs: 500 for MobileNetV2; 200 for others models
4) Other parameters are set in the ‘arguments.py’ file.

#### Fig. 1 The threat model of poisoning attacks
![image](https://github.com/Azhaoyaru/Shielding-Federated-Learning/blob/main/Threat%20model%20of%20poisoning%20attacks.png)

### Citing
If you use this code or it provides you with some reference value, please cite the paper：
  XXXXX

