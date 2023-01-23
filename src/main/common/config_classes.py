import configparser


class ConfigurableObject(object):
    def __init__(self):
        super().__init__()
        config = configparser.ConfigParser()
        config_file = config.read(['config.ini', '../config.ini', '../../config.ini', '../../../config.ini'])
        if len(config_file) == 0:
            raise Exception("config.ini file not found")
        self.config = config

    def get_config(self):
        return self.config
