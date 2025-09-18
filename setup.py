# Basic setup.py for Football Analytics Platform
from setuptools import setup, find_packages

setup(
    name="football-analytics-2025",
    version="1.0.0",
    description="AI-powered football analytics platform for player performance prediction",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "tensorflow>=2.13.0",
        "torch>=2.0.0",
        "fastapi>=0.100.0",
        "streamlit>=1.25.0",
        "plotly>=5.15.0",
        "prophet>=1.1.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "redis>=4.6.0",
        "python-dotenv>=1.0.0",
        "mlflow>=2.5.0",
        "pytest>=7.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "pre-commit>=3.3.0",
            "jupyter>=1.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)