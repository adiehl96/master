from Enumerations import ObservationModels


def talk(
    params, performance
):  # Tell the world about various performance stats, but why is the model talking to us?

    print("")
    print(f"UU : ", end="")
    if params["uu_observation_model"] == ObservationModels.Logit:
        print(
            f"AUC = {performance['auc']:.3f} : Error = {performance['classifier_error']:.3f}"
        )
    else:
        raise Exception("Only ObservationModels Logit Implemented")
