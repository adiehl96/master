import numpy as np


def flatten(the_lists):
    result = []
    for item in the_lists:
        if isinstance(item, list):
            result += item
        else:
            result.append(item)
    if any(isinstance(item, list) for item in result):
        result = flatten(result)
    return result
