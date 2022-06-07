# The goal is to move the entire `torchoptim` out into a standalone repository / library
# that people can pip install, since there is no Alpa / JAX related logic inside this folder.

from .adam import adam
