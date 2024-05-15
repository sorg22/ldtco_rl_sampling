import pandas as pd

tbl_ref = pd.read_csv("golden_route_predict_test1")
tbl_pred = pd.read_csv("test_pred_new_train_new_test.dat")

tbl_ref['routability'] = tbl_pred['pred']
tbl_ref.to_csv("golden_route_predict_test1_model_predict", index=False)
tbl_pred["code"] = tbl_ref["code"]
tbl_pred.to_csv("golden_route_predict_test1_model_predict_probability", index=False)