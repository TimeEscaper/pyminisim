from setuptools import setup


setup(
    name='pyminisim',
    version='0.1.0',
    packages=[
        'pyminisim',
        'pyminisim.core',
        'pyminisim.pedestrians',
        'pyminisim.pedestrians.assets',
        'pyminisim.robot',
        'pyminisim.sensors',
        'pyminisim.util',
        'pyminisim.visual',
        'pyminisim.visual.assets',
        'pyminisim.visual.util'
    ],
    install_requires=[
        'pygame',
        'numpy',
        'pillow',
        'numba',
        'scipy',
        'Cython'
        # 'git+https://github.com/sybrenstuvel/Python-RVO2.git'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
