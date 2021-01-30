import lumicks.pylake as lk


def modelsum(terms, model=False):
    """Intended for adding model terms onto a single model.

    Parameters
    ----------
    terms : list of pylake model objects
    model : pylake model obect
    """
    if model:
        if not terms:
            return model
        else:
            return modelsum(terms[1:], model + terms[0])
    else:
        return modelsum(terms[1:], terms[0])


def build_term(term):
    """Takes a string corresponing to a pylake model to return the function
    for that model.

    Parameters
    ----------
    term : str
        String corresponding to a key in the enclosed model dictionary.
    """
    modeldict = {"odijk": lk.odijk,
                 "inv_odijk": lk.inverted_odijk,
                 "marko_siggia_simple": lk.marko_siggia_simplified,
                 "inv_marko_siggia_simple": lk.inverted_marko_siggia_simplified,
                 "marko_siggia_dist": lk.marko_siggia_ewlc_distance,
                 "marko_siggia_force": lk.marko_siggia_ewlc_force,
                 "twistable_wlc": lk.twistable_wlc,
                 "inv_twistable_wlc": lk.inverted_twistable_wlc,
                 "free_jointed_chain": lk.freely_jointed_chain,
                 "inv_free_jointed_chain": lk.inverted_freely_jointed_chain,
                 "distance_offset": lk.distance_offset,
                 "force_offset": lk.force_offset}
    return modeldict[term]


def build_model(name, terms, model=False):
    """Constructs a composite model from a list of key values corresponding
    to pylake model objects.

    Parameters:
    -----------
    name : str
        Name for the model. This name will be prefixed to the model parameter names.
    terms : list of str
        Key values corresponding to pylake models.
    model : pylake model object
        Existing model object to add on to
    """
    if model:
        return model + modelsum([build_term(term)(name) for term in terms])
    else:
        return modelsum([build_term(term)(name) for term in terms])
