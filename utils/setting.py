import argparse

def parser_set():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0, help='coefficient of loss function')
    parser.add_argument("--gamma", type=int, default='10000', help='change training stage')
    parser.add_argument("--gpu", type=int, default=0, help="gpu ID")
    parser.add_argument("--model_name", type=str, default='LSTM')
    parser.add_argument("--transmission_num", type=int, default=8)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10000)
    args = parser.parse_args()
    return args


def log_string(log, string, p=True):  # p decide print
    log.write(string + '\n')
    log.flush()
    if p:
        print(string)