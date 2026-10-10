def get_model(model_name, args):
    name = model_name.lower()
    if name == "reprof":
        from models.reprof import Learner
        return Learner(args)
    elif name == "two_layer_fusion":
        from models.two_layer_fusion import Learner
        return Learner(args)
    elif name == "naive_average":
        from models.naive_average import Learner
        return Learner(args)
    elif name == "layer_mean":
        from models.layer_mean import Learner
        return Learner(args)
    elif name == "layer_concat":
        from models.layer_concat import Learner
        return Learner(args)
    elif name == "ranpac":
        from models.ranpac import Learner
        return Learner(args)
    else:
        assert 0
