from lib.models.monouni import MonoUNI


def build_model(cfg,mean_size):
    if cfg['type'] == 'monouni':
        return MonoUNI(backbone=cfg['backbone'], neck=cfg['neck'], mean_size=mean_size,cfg=cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
