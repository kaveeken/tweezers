import json

# turn this into sensible notebook cell

models = {"handles": {"terms": ["inv_odijk", "force_offset"],
                      "onto": False,
                      "invert": False},
          "protein": {"terms": ["inv_marko_siggia_simple"],
                      "onto": "handles",
                      "invert": "local"}}

estimates = {'handles/Lp': {'value': 18.76228992144659,
                            'upper_bound': 100,
                            'lower_bound': 0.0,
                            'bound_upr': True},
             'handles/Lc': {'value': 0.35161779501977486,
                            'lower_bound': 0.0,
                            'bound_upr': True},
             'handles/St': {'value': 265.14425878473025,
                            'lower_bound': 250,
                            'bound_upr': True},
             'kT': {'value': 4.11,
                    'upper_bound': 8.0,
                    'lower_bound': 0.0,
                    'bound_upr': True},
             'handles/f_offset': {'value': 0.032795025445327454,
                                  'upper_bound': 6,
                                  'lower_bound': -6,
                                  'bound_upr': False},
             'protein/Lp': {'value': 0.6000000000000009,
                            'upper_bound': 1.0,
                            'lower_bound': 0.6,
                            'bound_upr': False},
             'protein/Lc': {'value': 0.02525084386794315,
                            'lower_bound': 0.0,
                            'bound_upr': False}}

with open('config.json', 'w') as f:
    json.dump({"models": models, "estimates": estimates}, f, indent=4)
