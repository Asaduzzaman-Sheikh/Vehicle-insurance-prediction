from setuptools import setup, find_packages

setup()













































# Since you already have a complete pyproject.toml, you do not need any of the metadata in the image. Setuptools will automatically read your pyproject.toml and ignore the details in setup.py.
# You can replace the entire code in the image with just this:
# python

# from setuptools import setup

# setup()

# Why? Because your pyproject.toml already has the name, version, author, and dynamic dependencies (requirements.txt). This 3-line file just tells older tools to use setuptools.
# Option 2: The "Standalone" Approach

# If you want to use the style in the image (ignoring pyproject.toml for metadata) but make it correct for your project, it should look like this:
# python

# from setuptools import setup, find_packages

# Helper to read your requirements.txt
# def get_requirements():
#     with open("requirements.txt") as f:
#         return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# setup(
#     name="vehicle-insurance-premium-prediction",  # FIXED: Don't use "src"
#     version="0.1.0",
#     author="Asaduzzaman",                         # FIXED: Your name
#     author_email="asaduzzamansheikh777@gmail.com",# FIXED: Your email
#     description="A machine learning project to predict vehicle insurance premiums.",
#     packages=find_packages(),
#     install_requires=get_requirements(),          # FIXED: Added dependencies
# )

#  The Golden Rule for setup.py + pyproject.toml

# If you have both files in your project, pyproject.toml always wins.

#     If you run pip install ., pip reads pyproject.toml first.

#     If you use the image's code with your pyproject.toml, pip will use name="vehicle-insurance-premium-prediction" from the TOML, and it will ignore name="src" from the setup.py.

# Recommendation: Stick to Option 1. Just use setup() in the setup.py file, and let your pyproject.toml do all the heavy lifting. It is the modern, standard way to do things.
