import hashlib
import json
import logging
import shutil
import tempfile

logging.getLogger().setLevel(logging.INFO)


class Stage:
    def __init__(self, config):
        self.config = config
        self.hash = self.get_config_hash(config)
        self.tmpdir = tempfile.mkdtemp()

        if "cache" not in config or config["cache"]["mode"] == "disabled":
            load = False
            save = False
        else:
            load = config["cache"].get("load", False)
            if type(load) is str:
                load = self.__class__.__name__ == load
            elif type(load) is list:
                load = self.__class__.__name__ in load
            elif type(load) is dict:
                load = load.get(self.__class__.__name__, False)
                if type(load) is str:
                    self.hash = load
                    load = True

            save = config["cache"].get("save", False)
            if type(save) is str:
                save = self.__class__.__name__ == save
            elif type(save) is list:
                save = self.__class__.__name__ in save
            elif type(save) is dict:
                save = save.get(self.__class__.__name__, False)

            assert (
                    type(load) is bool and type(save) is bool
            ), f"Invalid cache config: {config['cache']}"

        logging.info(f"Stage {self.__class__.__name__}:"
                     f"hash: {self.hash}, "
                     f"tmpdir: {self.tmpdir}, "
                     f"load: {load}, "
                     f"save: {save}")

        if load:
            try:
                logging.info(f"Loading cache for {self.__class__.__name__}.")

                self.load()
                # Don't save again if we loaded
                save = False
            except Exception as e:
                logging.warning(
                    f"Loading cache not possible "
                    f"for {self.__class__.__name__}. "
                )
                logging.exception(e)
                self.generate()
        else:
            logging.info(f"Generating {self.__class__.__name__}.")

            self.generate()

        if save:
            logging.info(f"Saving {self.__class__.__name__}.")

            self.save()

    def __del__(self):
        try:
            shutil.rmtree(self.tmpdir)
        except AttributeError:
            pass

    def generate(self):
        raise NotImplementedError(
            f"generate() not implemented for {self.__class__.__name__}"
        )

    def load(self):
        raise NotImplementedError(
            f"load() not implemented for {self.__class__.__name__}"
        )

    def save(self):
        raise NotImplementedError(
            f"save() not implemented for {self.__class__.__name__}"
        )

    @classmethod
    def get_relevant_config(cls, config):
        raise NotImplementedError(
            f"get_relevant_config() not implemented for {cls.__name__}"
        )

    @classmethod
    def get_config_hash(cls, config):
        return hashlib.sha256(
            json.dumps(
                cls.get_relevant_config(config),
                default=lambda o: "<not serializable>",
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:6]
