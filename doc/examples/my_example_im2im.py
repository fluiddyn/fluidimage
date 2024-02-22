"""This is a very silly example of importable module where we define a
"filter" function im2im.

"""


def im2im(im_path):
    im, path = im_path
    print("in the function im2im...")
    return 2 * im, path
