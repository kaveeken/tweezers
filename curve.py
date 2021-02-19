import lumicks.pylake as lk
import numpy as np
from event_finding import get_first_trough_index, find_transitions, plot_events
from copy import deepcopy
from math import exp


class Curve_info:
    """ This class holds the data of an FD curve and methods used for analysis.
    """
    def __init__(self, identifier: str, ddata: np.ndarray, fdata: np.ndarray):
        self.identifier = identifier
        self.dist_data = ddata
        self.force_data = fdata

    
    def find_events(self, STARTING_FORCE: int = 0, CORDON: int = 10,
                    FIT_ON_RETURN: tuple[int] = (), DEBUG: bool = False):
        """ Identifies relevant events in the force data over time. From those
        events determines which parts (legs) of the data to mark for fitting.
        
        Arguments:
        - STARTING_FORCE: we do not mark for fitting data before the point where
        STARTING_FORCE is reached.
        - CORDON: we exclude from fitting datapoints within CORDON points
        before and after an event.
        - FIT_ON_RETURN: tuple describing an arbitrary part of the
        elaxation curve to mark for fitting. Should either be empty or contain
        two numbers describing how far after the return point and how many
        datapoints to include.
        - DEBUG: if something goes wrong with identifying the return point or
        stationary part of the curve, set this to True to print specific info.
        """
        if DEBUG:
            print(self.identifier)
        self.top = (get_first_trough_index(self.force_data, debug=DEBUG),
                    get_first_trough_index(self.force_data, last=True,
                                           debug=DEBUG))
        if self.top[1] - self.top[0] > 100:
            self.unfolds, self.threshold = \
                find_transitions(self.force_data,
                                 noise_estimation_window=self.top)
        else:
            self.unfolds, self.threshold = find_transitions(self.force_data)

        self.start = 0
        for index, force in enumerate(self.force_data):
            if force > STARTING_FORCE:
                start = index
                break

        events = [self.start, *self.unfolds, self.top[0]]
        self.legs = [slice(*[events[i] + CORDON, events[i+1] - CORDON])\
                     for i in range(len(events) - 1)]

        if FIT_ON_RETURN:
            self.legs.append(slice(top_window[-1] + FIT_ON_RETURN[0],
                                   top_window[-1] + sum(FIT_ON_RETURN)))


    def plot_events(self):
        return False

    
    def filter_tethers(self, model: lk.Model, estimates_dict: dict,
                       VERBOSE: bool = True):
        """ Tests for multiple tethers by comparing the fit of the raw estimates
        to changes made in those estimates that are meant to better describe
        the case of two tethers.
        Similarly tries to redescribe the system to better fit the two-tether
        case by halving force data and doubling distance data.
        
        Arguments:
        - model: Pylake model object.
        - estimates_dict: dictionary with the raw estimates as well as changed
        estimates for east test. Raw estimates have to be named 'original'.
        - VERBOSE: toggle to print out the test results immediately
        """
        fits = {test_id: lk.FdFit(model) for test_id in estimates_dict.keys()}
        handle_forces = self.force_data[self.legs[0]]
        handle_dists = self.dist_data[self.legs[0]]
        
        self.bics = {}
        for key, fit in fits.items():
            if key == 'half_force':
                fit.add_data(f'{self.identifier}_{test_id}',
                             handle_forces / 2, handle_dists)
                load_estimates(fit, estimates_dict['original'])
            elif key == 'double_dist':
                fit.add_data(f'{self.identifier}_{test_id}',
                             handle_forces, handle_dists * 2)
                load_estimates(fit, estimates_dict['original'])
            else:
                fit.add_data(f'{self.identifier}_{test_id}',
                             handle_forces, handle_dists)
                load_estimates(fit, estimates_dict[key])
            self.bics[key] = fit.bic

        self.bfactors = \
            {test_id: exp((self.bics['original'] - self.bics[test_id]) / 2) \
             for test_id in self.bics.keys() - 'original'}
        self.tether_tests = \
            {test_id: self.bics['original'] > self.bics[test_id] \
             for test_id in self.bics.keys() - 'original'}

        if VERBOSE:
            print(self.identifier, '\n', self.tether_tests)


    def initialize_fits(self, handles_model: lk.Model,
                        composite_model: lk.Model,
                        handle_estimates: dict[str, dict]):
        self.fits = [lk.FdFit(handles_model, composite_model)\
                     for unfold in self.unfolds]
        for fit in self.fits:
            fit[handles_model].add_data(f'{self.identifier}_handles_model',
                                        self.force_data[self.legs[0]],
                                        self.dist_data[self.legs[0]])
        load_estimates(self.fits[0], handle_estimates)
        self.fits[0].fit()
        self.fits[0]['handles/St'].fixed = True
        self.fits[0]['handles/Lp'].fixed = True
        self.fits[0]['handles/Lc'].fixed = True

        for fit in self.fits[1:]:
            load_estimates(fit, extract_estimates(self.fits[0]))

    def composite_fits(self, protein_estimates: dict[str, dict]):
        
            
