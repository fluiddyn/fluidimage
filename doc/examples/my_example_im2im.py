"""This is a very silly example of importable module where we define a
"filter" function im2im.

"""


def im2im(tuple_name_image):
    _, im = tuple_name_image
    print("in the function im2im...")
    return 2 * im
