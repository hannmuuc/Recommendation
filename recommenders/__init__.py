import importlib


def find_recommender_using_name(recommender_name):

    model_filename = "recommenders." + recommender_name + "_recommender"
    modellib = importlib.import_module(model_filename)
    recommender = None
    target_model_name = recommender_name.replace('_', '') + 'recommender'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            recommender = cls

    if recommender is None:
        print("In %s.py, there should be a subclass of BaseRecommender with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)
    return recommender


def get_option_setter(recommender_name):
    recommender_class = find_recommender_using_name(recommender_name)
    return recommender_class.modify_commandline_options


def create_recommender(opt):
    recommender = find_recommender_using_name(opt.recommender)
    instance = recommender(opt)
    print("recommender [%s] was created" % type(instance).__name__)
    return instance
