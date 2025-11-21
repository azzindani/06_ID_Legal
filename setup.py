"""
Setup script for Indonesian Legal RAG System

Install with: pip install -e .
Install with dev dependencies: pip install -e ".[dev]"
"""

from setuptools import setup, find_packages

setup(
    name="id-legal-rag",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
)
