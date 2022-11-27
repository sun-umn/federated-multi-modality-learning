import argparse
import sys
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner
import os

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("job_folder")
#     # parser.add_argument("--data_path", "-i", type=str, help="Input data_path")
#     parser.add_argument("--workspace", type=str, help="WORKSPACE folder", default="./workspace/")
#     parser.add_argument("--n_clients", "-n", type=int, help="number of clients")
#     # parser.add_argument("--client_list", "-c", type=str, help="client names list")
#     parser.add_argument("--threads", "-t", type=int, help="number of running threads", required=True)
#     # parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")
#     args = parser.parse_args()
#     return args


if __name__ == "__main__":
    """
    This is the main program when starting the NVIDIA FLARE server process.
    """

    if sys.version_info >= (3, 9):
        raise RuntimeError("Python versions 3.9 and above are not yet supported. Please use Python 3.8 or 3.7.")
    if sys.version_info < (3, 7):
        raise RuntimeError("Python versions 3.6 and below are not supported. Please use Python 3.8 or 3.7.")
    # args = vars(parse_args())
    # print(args)
    # args['client_names'] = "a,b"
    
    args = {
        "job_folder": "/home/le/cancerbert_ner/FL_simulation/fed_ner/medical_ner",
        "workspace": "/home/le/cancerbert_ner/FL_simulation/fed_ner/medical_ner/workspace/",
        "n_clients": 2,
        "threads": 2
    }
    
    simulator = SimulatorRunner(**args)
    if simulator.setup():
        simulator.run()
    os._exit(0)