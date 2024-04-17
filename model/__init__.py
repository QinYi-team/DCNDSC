def raw_model(model_param):
    from .DCNDSC import DCNDSC
    # model = LDMcnn(**model_param)
    model = DCNDSC(**model_param)
    return model
