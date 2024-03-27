"""Base class using matplotlib widgets

"""

from matplotlib.widgets import Button, TextBox


class AppMatplotlibWidgets:

    _buttons: dict
    _textboxes: dict

    def __init__(self):
        self._buttons = {}
        self._textboxes = {}

    def _create_button(self, fig, rect, text, func):
        ax = fig.add_axes(rect)
        button = Button(ax, text)
        button.on_clicked(func)
        self._buttons[text] = button
        return button

    def _create_text_box(self, fig, rect, name, func, initial):
        ax = fig.add_axes(rect)
        textbox = TextBox(ax, name, initial=initial)
        textbox.on_submit(func)
        self._textboxes[name] = textbox
        return textbox

    def get_textbox(self, key):
        return self._textboxes[key]
