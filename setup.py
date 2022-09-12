from distutils.core import setup

setup(
    name='ATAC',
    version='0.1.0',
    author='Ching-An Cheng',
    author_email='chinganc@microsoft.com',
    package_dir={'':'src'},
    python_requires='>=3.8',
    packages=['atac'],
    url='https://github.com/microsoft/ATAC',
    license='MIT LICENSE',
    description='ATAC code',
    long_description=open('README.md').read(),
    install_requires=[
        "garage==2021.3.0",
        "gym==0.17.2",],
    extras_require={
        'mujoco200': ["mujoco_py==2.0.2.8",  "d4rl @ git+https://github.com/chinganc/d4rl@master#egg=d4rl"],
        'mujoco210': ["d4rl @ git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl"]}
)
