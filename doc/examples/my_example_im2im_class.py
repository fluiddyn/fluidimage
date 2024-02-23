"""This is a very silly example of importable module where we define a
"filter" class Im2Im.

"""


class Im2Im:
    def __init__(self, arg0, arg1):
        self.arg0 = arg0
        self.arg1 = arg1

    def calcul(self, image_path):
        image, path = image_path
        print(f"in the function Im2Im.calcul (arg0={self.arg0})...")
        return 2 * image, path
