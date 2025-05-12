from typing import List
import os
from setuptools import setup, find_packages



def get_requirements()->List[str]:

    """
    This function wil return list of requirements 
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            #Read lines from the file
            lines=file.readlines()
            ## Process each line
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name="ml-project-iot",
    version="0.0.1",
    author="Jagath",
    author_email="jagath1371@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)