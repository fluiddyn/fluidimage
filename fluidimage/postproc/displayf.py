import pylab


def displayf(X, Y, U=None, V=None, background=None, *args):

    if background is not None:
        pylab.pcolor(X, Y, background)
    if U is not None:
        pylab.quiver(X, Y, U, V)
