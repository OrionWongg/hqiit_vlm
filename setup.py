from setuptools import setup
import os
from glob import glob

package_name = 'hqiit_vlm'  # 替换为你的功能包名称

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
    # 新增Python第三方库依赖
    install_requires=['setuptools', 'requests', 'pyyaml', 'opencv-python'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@xxx.com',
    description='ROS 2 VLM Node Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    # 关键：声明节点入口（让ros2 run能找到节点）
    entry_points={
        'console_scripts': [
            'vlm_pub_node = hqiit_vlm.vlm_pub:main',
        ],
    },
)