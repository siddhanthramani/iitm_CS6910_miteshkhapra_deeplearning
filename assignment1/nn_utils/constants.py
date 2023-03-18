# All global constants can be defined here
# Enclosed in a class - best practice - to prevent errors in multiprocessing for different user inputs

class global_constants():
    def __init__(self, epsilon = 1e-8):
        self.epsilon = 1e-8