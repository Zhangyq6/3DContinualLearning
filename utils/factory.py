def get_model(args, model_config = None):
    if args.model_name == "ga":
        from models.ga import Learner
    elif args.model_name == "l2p":
        from models.l2p import Learner
    return Learner(args)
