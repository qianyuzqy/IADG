import logging


class Logger(object):

    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, name='logger',
                 log_path=None,
                 level='info',
                 fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(name)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(fmt))
        logging.root.addHandler(console_handler)
        logging.root.setLevel(self.level_relations.get(level))
        if log_path:
            file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
            file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
            file_handler.setFormatter(file_formatter)
            logging.root.addHandler(file_handler)

    @property
    def debug(self):
        return self.logger.debug

    @property
    def info(self):
        return self.logger.info

    @property
    def warning(self):
        return self.logger.warning

    @property
    def error(self):
        return self.logger.error

    @property
    def critical(self):
        return self.logger.critical


if __name__ == '__main__':
    log = Logger('all.log', level='info')
    log.debug('debug')
    log.info('info')
    log.warning('warning')
    log.error('error')
    log.critical('critical')
