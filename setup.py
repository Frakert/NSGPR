from setuptools import setup, find_packages


setup(
    name="NSGPR", 
    version="0.1.0",  # Update version as needed
    description="A Gaussian Process library with stationary and non-stationary models",
    packages=find_packages(where="src"),  # Finds all packages under `src/`
    package_dir={"": "src"},  # Tells setuptools where to look for packages
    include_package_data=True
)
