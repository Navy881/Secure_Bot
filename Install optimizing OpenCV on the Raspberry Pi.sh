# 1. Delete both Wolfram Engine and LibreOffice to reclaim ~1GB of space on your Raspberry Pi:
sudo apt-get purge wolfram-engine
sudo apt-get purge libreoffice*
sudo apt-get clean
sudo apt-get autoremove

# 2. Install dependencies
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk2.0-dev libgtk-3-dev
sudo apt-get install libcanberra-gtk*
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python2.7-dev python3-dev 

# 3. Download the OpenCV source code
# 3.3.0 - version (last 3.4.6)
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/3.3.0.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.3.0.zip
unzip opencv_contrib.zip

# 4. Create your Python virtual environment and install NumPy	
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo python3 get-pip.py
sudo pip install virtualenv virtualenvwrapper
sudo rm -rf ~/.cache/pip

# 5. Update ~/.profile file
echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.profile
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.profile
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.profile
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.profile

# 6. Reload ~/.profile file
source ~/.profile

# 7. Create Python 3 virtual environment
mkvirtualenv cv -p python3

# 8. Install NumPy into the Python virtual environment
pip install numpy

# 9. Run cv virtual environment
workon cv

# 10. Build OpenCV
cd ~/opencv-3.3.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.0/modules \
    -D ENABLE_NEON=ON \
    -D ENABLE_VFPV3=ON \
    -D BUILD_TESTS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=OFF ..

# 11. Increasing swap space
sudo nano /etc/dphys-swapfile
# Correct CONF_SWAPSIZE=100 -> CONF_SWAPSIZE=1024
# Ctrl-O -> Enter -> Ctrl-X

# 12. Restart the swap service
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start

# 13. Make OpenCV
sudo make -j4
sudo make install
sudo ldconfig

# 14. Decreasing swap space
sudo nano /etc/dphys-swapfile
# Correct CONF_SWAPSIZE=1024 -> CONF_SWAPSIZE=100
# Ctrl-O -> Enter -> Ctrl-X

# 15. Restart the swap service
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start

# 16. Check exist .so file
ls -l /usr/local/lib/python3.5/site-packages
# total 1852
# -rw-r--r-- 1 root staff 1895932 Mar 20 21:51 cv2.cpython-34m.so

# 17. Rename .so file
cd /usr/local/lib/python3.5/site-packages/
sudo mv cv2.cpython-35m-arm-linux-gnueabihf.so cv2.so

# 18. Sym-link our OpenCV bindings into the cv virtual environment for Python 3.5
cd ~/.virtualenvs/cv/lib/python3.5/site-packages/
ln -s /usr/local/lib/python3.5/site-packages/cv2.so cv2.so

# 19. Testing OpenCV install
source ~/.profile 
workon cv
python
# >>> import cv2
# >>> cv2.__version__
# '3.3.0'
# >>>

# ! Everytime run
source ~/.profile 
workon cv