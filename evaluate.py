# evaluate max F-measure, S-measure and MAE

import os
from evaluator.evaluator import evaluate_dataset
from evaluator.utils import write_doc
from evaluator.log import create_logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='YYNet')
parser.add_argument("--test_path", type=str, default='/hy-tmp/dataset/test/gts/')
parser.add_argument("--prediction_path", type=str, default='./prediction1/')
opt = parser.parse_args()


def evaluate(roots, doc_path, num_thread, pin):
    datasets = roots.keys()

    results = evaluate_dataset(roots=roots[dataset], 
                                dataset=dataset,
                                batch_size=1, 
                                num_thread=num_thread, 
                                demical=True,
                                suffixes={'gt': '.png', 'pred': '.png'},
                                pin=pin)

    # Save evaluation results.
    content = '{}:\n'.format(dataset)
    #content += 'max-Fmeasure={}'.format(results['max_f'])
    content += 'max-Fmeasure={} '.format(results['max_f'])
    content += 'max-Emeasure={} '.format(results['max_e'])
    content += ' '
    content += 'Smeasure={}'.format(results['s'])
    content += ' '
    content += 'MAE={}\n'.format(results['mae'])
    write_doc(doc_path, content)

    return content


# ------------- end -------------

if __name__ == "__main__":
    logger = create_logger('evaluate')

    eval_device = '0'
    eval_doc_path = './eva.txt'
    eval_num_thread = 10

    logger.info('eval_doc_path:{}\neval_num_thread:{}'.format(eval_doc_path, eval_num_thread))
    logger.info('model name:{}'.format(opt.model_name))

    # An example to build "eval_roots".
    eval_roots = dict()
    test_datasets = ['CoCA', 'CoSal2015','CoSOD3k'] 
    logger.info(test_datasets)

    for dataset in test_datasets:
        print(dataset)
        roots = {'gt': opt.test_path + dataset + '/',
                'pred': opt.prediction_path + dataset + '/'}
        eval_roots[dataset] = roots
        os.environ['CUDA_VISIBLE_DEVICES'] = eval_device
        
        content = evaluate(roots=eval_roots, 
                doc_path=eval_doc_path,
                num_thread=eval_num_thread,
                pin=False)

        logger.info('pred:{}'.format(roots['pred']))

        logger.info(content)
        print(content)



