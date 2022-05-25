import pickle
from Interface.SetupObjects import setup_objects


def rfm_experiment(params=None):
    params = {} if params is None else params

    save_filename = params.get("save_filename", "DefaultSaveFile.pickle")
    save_traces = params.get("save_filename", False)
    save_prediction = params.get("save_filename", False)

    rfm, mcmc = setup_objects(params)

    mcmc.sample()

    performance = rfm.performance(False, mcmc.predictions_average)
    print("\n **** Average performance **** \n")
    rfm.talk(performance)
    print("")

    if save_traces:
        pickle.dump([performance, rfm, mcmc], open(save_filename, "wb"))
    else:
        if save_prediction:
            prediction = mcmc.predictions_average
            pickle.dump([performance, prediction], open(save_filename, "wb"))
        else:
            pickle.dump([performance], open(save_filename, "wb"))
