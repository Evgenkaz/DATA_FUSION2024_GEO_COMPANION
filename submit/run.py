#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
import gc
from pathlib import Path
from typing import Annotated
from warnings import simplefilter
from model_vtb import count_transform,load_models
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import Ridge
from xgboost import XGBClassifier
import typer
import joblib
from typer import Option
import pickle
import json

app = typer.Typer()

def main(
  hexses_target_path: Annotated[
    Path, Option('--hexses-target-path', '-ht', dir_okay=False, help='Список локаций таргета', show_default=True, exists=True)
  ] = 'hexses_target.lst',
  hexses_data_path: Annotated[
    Path, Option('--hexses-data-path', '-hd', dir_okay=False, help='Список локаций транзакций', show_default=True, exists=True)
  ] = 'hexses_data.lst',
  input_path: Annotated[
    Path, Option('--input-path', '-i', dir_okay=False, help='Входные данные', show_default=True, exists=True)
  ] = 'moscow_transaction_data01.parquet',
  output_path: Annotated[
    Path, Option('--output-path', '-o', dir_okay=False, help='Выходные данные', show_default=True)
  ] = 'output.parquet',
):
    with open(hexses_target_path, "r") as f:
        hexses_target = [x.strip() for x in f.readlines()]
    with open(hexses_data_path, "r") as f:
        hexses_data = [x.strip() for x in f.readlines()]


    transactions = pd.read_parquet(input_path)

    test_data = count_transform(transactions, hexses_data)

    X_test = test_data.drop('customer_id', axis=1).fillna(0)
    submit = test_data[['customer_id']]

    model_0 = joblib.load(f'models/model_0.pkl')
    model_1 = joblib.load(f'models/model_1.pkl')
    model_2 = joblib.load(f'models/model_2.pkl')
    model_3 = joblib.load(f'models/model_3.pkl')
    model_4 = joblib.load(f'models/model_4.pkl')
    model_5 = joblib.load(f'models/model_5.pkl')
    model_6 = joblib.load(f'models/model_6.pkl')

    pred = (model_0.predict_proba(X_test)+model_1.predict_proba(X_test)+model_2.predict_proba(X_test)\
            +model_3.predict_proba(X_test)+model_4.predict_proba(X_test)+model_5.predict_proba(X_test)\
            +model_6.predict_proba(X_test))/7
    submit[hexses_target]=pred
 

    submit.to_parquet(output_path)

if __name__ == '__main__':
  typer.run(main, )
