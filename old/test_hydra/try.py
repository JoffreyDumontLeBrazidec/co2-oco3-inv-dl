from omegaconf import OmegaConf
import tempfile

conf = OmegaConf.create({"foo": 10, "bar": 20, 123: 456})
OmegaConf.save(config=conf, f="omega.yaml")
