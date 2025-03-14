from setuptools import setup, find_packages

setup(
    name="nes_sprite_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pillow",
        "openai",
        "anthropic",
        "google-generativeai",
        "flask",
    ],
    entry_points={
        "console_scripts": [
            "nes-sprite-generator=nes_sprite_generator.cli:main",
        ],
    },
)