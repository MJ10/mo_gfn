from torch_seq_moo.metrics.r2 import r2_indicator_set
from torch_seq_moo.metrics.hsr_indicator import HSR_Calculator
from pymoo.factory import get_performance_indicator
import numpy as np

def get_all_metrics(solutions, eval_metrics, **kwargs):
    """
    This method assumes the solutions are already filtered to the pareto front
    """
    
    metrics = {}
    if "hypervolume" in eval_metrics and "hv_ref" in kwargs.keys():
        hv_indicator = get_performance_indicator('hv', ref_point=kwargs["hv_ref"])
        # `-` cause pymoo assumes minimization
        metrics["hypervolume"] = hv_indicator.do(-solutions)
    
    if "r2" in eval_metrics and "r2_prefs" in kwargs.keys() and "num_obj" in kwargs.keys():
        metrics["r2"] = r2_indicator_set(kwargs["r2_prefs"], solutions, np.ones(kwargs["num_obj"]))
    
    if "hsri" in eval_metrics and "num_obj" in kwargs.keys():
        # class assumes minimization so transformer to negative problem
        hsr_class = HSR_Calculator(lower_bound=-np.ones(kwargs["num_obj"]) - 0.1, upper_bound=np.zeros(kwargs["num_obj"]) + 0.1)
        # try except cause hsri can run into divide by zero errors 
        try:
            metrics["hsri"], x = hsr_class.calculate_hsr(-solutions)
        except:
            metrics["hsri"] = 0.
        try:
            metrics["hsri"] = metrics["hsri"] if type(metrics["hsri"]) is float else metrics["hsri"][0]
        except:
            metrics["hsri"] = 0.
    return metrics