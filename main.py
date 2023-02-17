import warnings
from datetime import datetime

import torch
import pandas as pd
from tqdm.std import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

warnings.filterwarnings('ignore')

FIELDS = {
    'product_species': '제품_종',
    'product_origin': '원산지',
    'rawmaterial_origin': '원료',
    'product_manufacturer': '제조사',
    'product_brand': '브랜드',
    'sales_unit': '판매단위',
    'product_storage_method': '저장방식',
    'product_grade': '제품등급',
    'delivery': '배달여부',
    'product_attribute': '제품특성',
    'product_container': '포장용기',
    'bundle_count': '개당갯수',
    'item_count': '포장갯수',
    'total_scale': '총무게',
    'bundle_scale': '번들무게',
    'item_scale': '개당무게',
    'spec': '제품사양'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('AgileGrowth/food-parser-t5-tiny-cased')
model = T5ForConditionalGeneration.from_pretrained('AgileGrowth/food-parser-t5-tiny-cased').to(device)


@torch.no_grad()
def predict(rawname: str) -> dict:
    input_data = [f"{rawname}</s>{field}" for field in FIELDS.values()]
    input_encodings = tokenizer.batch_encode_plus(input_data, padding=True, return_tensors='pt')
    input_ids = input_encodings['input_ids'].to(device)

    outputs = model.generate(input_ids)
    pred_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = {k: v if v != 'X' else '' for k, v in zip(FIELDS.keys(), pred_answers)}

    return results


def evaluate(data: dict) -> dict:
    rawname = data['rawname']
    answers = [data[field] for field in FIELDS.keys()]
    pred_answers = predict(rawname)

    results = {field: {} for field in FIELDS.keys()}
    for field, answer, pred_answer in zip(FIELDS.keys(), answers, pred_answers.values()):

        if (answer is None) or pd.isna(answer):
            answer = ''
        answer = str(answer)

        results[field]['answer'] = answer
        results[field]['correct'] = 0

        if answer == '':
            results[field]['correct'] = int(answer == pred_answer)
        else:
            answer = ''.join(sorted(answer))
            pred_answer = ''.join(sorted(pred_answer))
            results[field]['correct'] = int(answer == pred_answer)

    return results


def evaluate_all(list_of_data: list) -> None:
    results = {
        field: dict.fromkeys(('total', 'count', 'x_count', 'success', 'x_success', 'fail', 'x_fail'), 0)
        for field in FIELDS.keys()
    }

    for data in tqdm(list_of_data):
        eval_result = evaluate(data)
        for field in eval_result.keys():
            results[field]['total'] += 1

            if eval_result[field]['answer'] == '':
                results[field]['x_count'] += 1
                results[field]['x_success' if eval_result[field]['correct'] else 'x_fail'] += 1
            else:
                results[field]['count'] += 1
                results[field]['success' if eval_result[field]['correct'] else 'fail'] += 1

    df = pd.DataFrame(results).T
    df['accuracy'] = df['success'] / df['count']
    df['x_accuracy'] = df['x_success'] / df['x_count']
    df['total_accuracy'] = (df['success'] + df['x_success']) / df['total']

    df.loc['average', 'accuracy'] = df['accuracy'].mean()
    df.loc['average', 'x_accuracy'] = df['x_accuracy'].mean()
    df.loc['average', 'total_accuracy'] = df['total_accuracy'].mean()

    save_fp = datetime.now().isoformat() + '.csv'
    df.to_csv(path_or_buf=save_fp)


if __name__ == '__main__':
    """
    df = pd.read_excel('data/x-x/상품분석기_샘플데이터_3차_수정.xlsx',
                       sheet_name='정답지(변경후)', dtype='str')
    evaluate_all(df.T.to_dict().values())
    """

    """
    evaluate_all([
        {
            'rawname': '돈육(전지,냉장,통덩어리,국산),Kg,1kg',
            'product_species': '돼지고기',
            'product_origin': '국내산',
            'rawmaterial_origin': None,
            'product_manufacturer': None,
            'product_brand': None,
            'sales_unit': 'kg',
            'bundle_count': None,
            'item_count': None,
            'total_scale': '1kg',
            'bundle_scale': None,
            'item_scale': '1kg',
            'product_storage_method': '냉장',
            'product_grade': None,
            'delivery': None,
            'product_container': None,
            'spec': None,
            'product_attribute': '전지,통덩어리'
        },
    ])
    """
