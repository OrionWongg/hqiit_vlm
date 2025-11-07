from setuptools import setup
import os
from glob import glob

package_name = 'hqiit_vlm'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'requests',
        'pyyaml',
        'opencv-python',
        'opencv-python-headless',
        'numpy',
        'pillow'
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@xxx.com',
    description='ROS 2 VLM Node Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dino_vlm_pub_node = hqiit_vlm.dino_vlm_pub:main',
            'yolo_vlm_pub_node = hqiit_vlm.yolo_vlm_pub:main',
        ],
    },
)