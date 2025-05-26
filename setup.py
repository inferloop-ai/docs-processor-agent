from setuptools import setup, find_packages

setup(
    name="document-processor-agent",
    version="1.0.0",
    description="AI-powered document processing system with RAG and reflexion",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        # Dependencies will be loaded from requirements.txt
    ],
)
