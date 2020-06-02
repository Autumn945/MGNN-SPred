import copy
import json
import sys, os
import time

from tqdm import tqdm as h_tqdm


# data_dir = './data'

def check_path(home):
    if not os.path.isdir(home):
        print(f'Not found dir: {home}, creating it.')
        os.mkdir(home)

def red_str(s, tofile=False):
    s = str(s)
    if tofile:
        # s = f'**{s}**'
        pass
    else:
        s = f'\033[1;31;40m{s}\033[0m'
    return s


def get_time_str():
    return time.strftime('%Y-%m-%d_%H.%M.%S') + str(time.time() % 1)[1:6]


def save_json(data, fn, sort_keys=True, indent=None):
    with open(fn, 'w') as f:
        json.dump(data, f, sort_keys=sort_keys, indent=indent)


def load_json(fn):
    with open(fn, 'r') as f:
        return json.load(f)


def gen_data_from_jsonl(fn):
    with open(fn, 'r') as f:
        for line in f:
            data = json.loads(line)
            yield data


class Logger:
    def __init__(self, fn, verbose=1):
        self.pre_time = time.time()
        self.fn = fn
        self.verbose = verbose

    def __str__(self):
        return self.fn

    def log(self, s='', end='\n', red=False):
        s = str(s)
        if self.verbose == 1:
            p = red_str(s) if red else s
            print(p, end=end)
        elif self.verbose == 2:
            p = red_str(s, tofile=True) if red else s
            print(p, end=end)
        now_time = time.time()
        s = s + end
        if now_time - self.pre_time > 30 * 60:
            s = get_time_str() + '\n' + s
            self.pre_time = now_time
        with open(self.fn, 'a') as f:
            fs = red_str(s, tofile=True) if red else s
            f.write(fs)
        sys.stdout.flush()


class Graph:
    def __init__(self, min_cnt=1):
        self.min_cnt = min_cnt
        self.edge_cnt = {}
        self.adj = {}
        self._nb_edges = 0

    def add_edge(self, a, b):
        e = (a, b)
        self.edge_cnt.setdefault(e, 0)
        self.edge_cnt[e] += 1

        if self.edge_cnt[e] == self.min_cnt:
            self.adj.setdefault(a, [])
            self.adj[a].append(b)
            self._nb_edges += 1

    def has_edge(self, a, b):
        cnt = self.edge_cnt.get((a, b), 0)
        return cnt >= self.min_cnt

    def get_edges(self):
        edges = sorted([(a, b) for (a, b), cnt in self.edge_cnt.items() if cnt >= self.min_cnt])
        return edges

    def get_adj(self, a):
        return self.adj.get(a, [])

    def nb_edges(self):
        return self._nb_edges


# noinspection PyUnresolvedReferences
class Object(object):
    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def vars(self):
        return self.__dict__

    def keys(self):
        return sorted(self.__dict__.keys())

    def values(self, keys=None):
        if keys is None:
            keys = self.keys()
        return [self.__dict__[k] for k in keys]

    def copy(self):
        return copy.deepcopy(self)

    def get(self, name, d=None):
        return self.__dict__.get(name, d)

    def set(self, name, value):
        self.__dict__[name] = value

    def filter(self, *keys):
        ret = Object()
        for k in keys:
            ret.set(k, self.get(k))
        return ret

    def update(self, **args):
        self.__dict__.update(args)
        return self

    def setdefault(self, **args):
        for k, v in args.items():
            if k not in self.__dict__:
                self.__dict__[k] = v
        return self

    def prt_json(self):
        d = {}
        for k, v in self.__dict__.items():
            if type(v) not in (dict, list, int, float, bool, str):
                v = str(v)
            d[k] = v
        return json.dumps(d, indent=2, sort_keys=True)

    def prt_line(self):
        return json.dumps(self.__dict__, sort_keys=True)

    def __str__(self):
        lines = []
        for k in sorted(self.keys()):
            lines.append(f'{k}: {self.get(k)}')
        return '\n'.join(lines)

    def __repr__(self):
        return self.__str__()

    def format_prt(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))

class AsyncStream:
    def __init__(self, stream, max_q_size=1, mode='MT'):
        self.stream = stream
        self.close = False
        if mode == 'MP':
            from multiprocessing import Process, Queue
            self.q = Queue(max_q_size)
            self.p = Process(target=self.task)
            self.p.daemon = True
            self.p.start()
        else:
            import threading
            from queue import Queue
            self.q = Queue(max_q_size)
            self.p = threading.Thread(target=self.task)
            self.p.daemon = True
            self.p.start()

    def task(self):
        for d in self.stream:
            self.q.put(d)
        self.q.put(None)

    def generator(self):
        while True:
            d = self.q.get()
            if d is None:
                break
            yield d

class my_tqdm:
    def __init__(self, desc, total, leave):
        self.desc = desc
        self.total = total
        self.cnt = 0
        self.leave = leave
        if self.leave:
            print(f'>>> begin {self.desc}...')
        self.bt = time.time()

    def update(self, n):
        self.cnt += n

    def close(self):
        t = time.time() - self.bt
        v = self.cnt / t
        if self.leave:
            print(f'--- {self.desc} end, cnt: {self.cnt}, time: {t:.0f}s, v: {v:.2f}it/s')


def tqdm(verbose=None, desc='tqdm', leave=True, total=None):
    if verbose is None:
        verbose = args.get('verbose', 1)
    if verbose == 1:
        return h_tqdm(desc=desc, total=total, leave=leave, ncols=90, ascii=True)
    return my_tqdm(desc=desc, total=total,  leave=leave)

args = Object()
run_time_dir = 'run_time'
check_path(run_time_dir)

data_dir = f'{run_time_dir}/data'
log_dir = f'{run_time_dir}/log'
save_dir = f'{run_time_dir}/save'

check_path(data_dir)
check_path(log_dir)
check_path(save_dir)



def main():
    pass


if __name__ == '__main__':
    main()
