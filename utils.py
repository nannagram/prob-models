import sys

def read_add_paths(path):
    with open(path, 'r') as f:
        paths = f.read().splitlines()
    for path in paths:
        sys.path.append(path)
    return paths
