import gc
import torch

from changeByLayer import ChangeByLayer
from entropyByLayer import entropyByLayer
from HeadwiseDivergence import HeadwiseDivergence
from layerWiseDifference import LayerwiseDifference
from layerwiseResidualDifference import LayerwiseResidualDifference
from LogitLensAnalysis import LogitLensAnalysis


#Models = [
#    "Qwen/Qwen2.5-7B-Instruct",
#    "allenai/Olmo-3-7B-Instruct",
#    "swiss-ai/Apertus-8B-Instruct-2509",
#    "Qwen/Qwen2-0.5B-Instruct",
#    "meta-llama/Meta-Llama-3-8B-Instruct"
#]
Models = [
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

experiments = [
    ChangeByLayer,
    entropyByLayer,
    HeadwiseDivergence,
    LayerwiseDifference,
    LayerwiseResidualDifference,
    LogitLensAnalysis,
]

datafeeds = ["3", "5", "not_enough_info", "two_groups"]


def cleanup_experiment(exp):
    if exp is None:
        return

    if hasattr(exp, "model") and exp.model is not None:
        del exp.model
        exp.model = None

    if hasattr(exp, "tokenizer") and exp.tokenizer is not None:
        del exp.tokenizer
        exp.tokenizer = None

    del exp

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


for model in Models:
    for ExperimentClass in experiments:
        exp = None

        try:

            exp = ExperimentClass(Model_name=model)

            for datafeed in datafeeds:

                exp.run_analysis(manipulation_type=datafeed)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in {ExperimentClass.__name__} with {model}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            cleanup_experiment(exp)
