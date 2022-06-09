#!/bin/sh
cd /usr/local/lib
sudo mkdir python3.7
cd python3.7
sudo mkdir dist-packages
cd ~
git clone https://github.com/Freenove/Freenove_RPI_WS281x_Python.git
cd ~/Freenove_RPI_WS281x_Python
sudo python setup.py install
echo "The installation is complete!"
cd ~/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code
sudo python setup.py
sudo chmod u+x ./get_pi_requirements.sh
sudo ./get_pi_requirements.sh
