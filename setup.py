from setuptools import setup


setup(
    name='pyminisim',
    version='0.1.0',
    packages=[
        'pyminisim',
        'pyminisim.core',
        'pyminisim.pedestrians',
        'pyminisim.robot',
        'pyminisim.sensors',
        'pyminisim.visual',
        'pyminisim.visual.assets',
        'pyminisim.visual.util'
    ],
    install_requires=[
        'pygame',
        'numpy',
        'pillow',
        'numba',
        'scipy'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
