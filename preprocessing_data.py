import sys, os, time
from tqdm import tqdm

data_home = 'run_time/data'

def fx_ts(all_ts):
    max_ts, min_ts = max(all_ts), min(all_ts)
    dt = max_ts - min_ts
    print('max, min, dt')
    print(min_ts, max_ts)
    print(dt, dt / (24 * 3600))



def yc_preprocess():
    path = 'yc'
    # N = 1000000
    N = 0

    all_ts = []


    sid2vid_list = {}

    # Session ID,Timestamp,Item ID,Price,Quantity
    # 420374,2014-04-06T18:44:58.314Z,214537888,12462,1
    cnt = 0
    pbar = tqdm(desc='read buys')
    with open(f'{data_home}/{path}/yoochoose-buys.dat', 'r') as f:
        for line in f:
            pbar.update(1)
            cnt += 1
            if N > 0 and cnt > N: break
            line = line[:-1]
            sid, ts, vid, _, _ = line.split(',')
            ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S')))

            all_ts.append(ts)
            
            sid2vid_list.setdefault(sid, [])
            sid2vid_list[sid].append([vid, 0, ts])
    pbar.close()

    fx_ts(all_ts)
    # return


    # session_id,timestamp,item_id,category
    # 1,2014-04-07T10:51:09.277Z,214536502,0
    cnt = 0
    pbar = tqdm(desc='read clicks')
    with open(f'{data_home}/{path}/yoochoose-clicks.dat', 'r') as f:
        f.readline()
        for line in f:
            pbar.update(1)
            cnt += 1
            if N > 0 and cnt > N: break
            line = line[:-1]
            sid, ts, vid, _ = line.split(',')
            ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S')))

            sid2vid_list.setdefault(sid, [])
            sid2vid_list[sid].append([vid, 1, ts])
    pbar.close()



    # return

    for sid in sid2vid_list:
        sid2vid_list[sid] = sorted(sid2vid_list[sid], key=lambda x: x[-1])

    n = len(sid2vid_list)
    yc = sorted(sid2vid_list.items(), key=lambda x: x[1][-1][-1])

    frac = 1

    n_part = n // frac
    yc_part = yc[-n_part:]

    out_path = f'{path}_1_{frac}'
    os.mkdir(f'{data_home}/{out_path}')
    with open(f'{data_home}/{out_path}/data.txt', 'w') as f:
        for sid, vid_list in yc_part:
            vid_list = ','.join(map(lambda vid: ':'.join(map(str, vid)), vid_list))
            sess = ' '.join([sid, vid_list])
            f.write(sess + '\n')
        
    print(len(yc_part))

    print(yc[-1])


def main():
    print('hello world, preprocessing_data.py')
    yc_preprocess()

if __name__ == '__main__':
    main()

