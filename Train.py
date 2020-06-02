import time

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import utils
from utils import args, tqdm


class Train:
    def __init__(self, Model, data):
        self.data = data
        self.build_model(Model)
        self.has_train = False

    def build_model(self, Model):
        self.model = Model(self.data)

    def train(self):
        brk = 0
        best_vali = 0
        data_generator = self.data.gen_train_batch_for_train(args.batch_size)
        for ep in range(args.epochs):
            pbar = tqdm(total=args.nb_vali_step, desc='training', leave=False)
            loss = []
            t0 = time.time()
            for _ in range(args.nb_vali_step):
                data = next(data_generator)
                _loss = self.model.fit(data)

                loss.append(_loss)
                pbar.update(1)
            pbar.close()
            train_time = time.time() - t0

            vali_v, vali_str = self.metric('vali')
            if vali_v > best_vali:
                brk = 0
                best_vali = vali_v
                self.model.save()
            else:
                brk += 1
            red = (brk == 0)

            msg = f'#{ep + 1}/{args.epochs} loss: {np.mean(loss):.5f}, brk: {brk}, vali: {vali_str}'
            if args.show_test and args.nb_test > 0:
                _, test_str = self.metric('test')
                msg = f'{msg}, test: {test_str}'
            vali_time = time.time() - t0 - train_time
            msg = f'{msg}, time: {train_time:.0f}s,{vali_time:.0f}s'

            args.log.log(msg, red=red)

            if ep < 60:
                brk = 0
            if brk >= args.early_stopping:
                break
            self.has_train = True

    def final_test(self):
        self.model.restore()
        _, ret = self.metric('test')
        return ret


    def metric(self, name):
        data_gen = self.data.gen_metric_batch(name, batch_size=256)
        pred_list, true_vid = self.topk(data_gen)

        pred_list = np.array(pred_list)
        true_vid = np.expand_dims(np.array(true_vid), -1)

        k = 100
        acc_ar = (pred_list == true_vid)[:, :k]  # [BS, K]
        acc = acc_ar.sum(-1)

        rank = np.argmax(acc_ar[:, :k], -1) + 1
        mrr = (acc / rank).mean()
        ndcg = (acc / np.log2(rank + 1)).mean()

        acc = acc.mean()
        # print(acc_ar)
        # print(mrr)
        # input()
        acc *= 100
        mrr *= 100
        ndcg *= 100
        ret = acc
        # ret = acc + mrr * 10 + ndcg * 5
        # ret = acc + mrr + ndcg
        # return ret, 'HR%:{:.3f},MRR%:{:.4f},NDCG%:{:.4f}'.format(acc, mrr, ndcg)
        return ret, '{:.3f},{:.4f},{:.4f}'.format(acc, mrr, ndcg)

    def topk(self, data_gen):
        pred_list = []
        true_vid = []
        cnt = 0
        pbar = tqdm(desc='predicting...', leave=False)
        for data in data_gen:
            v, i = self.model.topk(data)
            pred_list.extend(i.tolist())
            true_vid.extend(data[2])
            pbar.update(1)
            cnt += 1
            if args.run_test and cnt > 3:
                break
        pbar.close()
        return pred_list, true_vid


def main():
    print('hello world, Train.py')


if __name__ == '__main__':
    main()
