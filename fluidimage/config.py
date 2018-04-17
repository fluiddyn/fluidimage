
import os

try:  # python 3
    from configparser import ConfigParser
except:  # python 2.7
    from ConfigParser import ConfigParser


def get_config():
    config = ConfigParser()

    home = os.path.expanduser("~")
    path_config = os.path.join(home, ".fluidimagerc")

    if os.path.exists(path_config):
        config.read(path_config)

    config_dict = {}
    for section in config.sections():
        section_dict = {}
        for option in config.options(section):
            value = config.get(section, option)
            if value.lower in ["true", "false"]:
                value = value == "true"
            section_dict[option] = value
        config_dict[section] = section_dict

    return config_dict
