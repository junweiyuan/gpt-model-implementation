from setuptools import setup, find_packages

setup(
    name="mobile-gui-agent",
    version="0.1.0",
    description="Mobile GUI Agent Framework with Reasoning VLMs",
    author="Mobile GUI Agent Team",
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.18.0",
        "google-generativeai>=0.3.0",
        "openai>=1.12.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.8",
)
