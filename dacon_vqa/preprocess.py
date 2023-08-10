import pandas as pd

train_df = pd.read_csv("/home/hwlee/dacon/imgQA/train.csv")
test_df = pd.read_csv("/home/hwlee/dacon/imgQA/test.csv")

train_df["img_path"] = train_df["image_id"].apply(lambda x: f"/home/hwlee/dacon/imgQA/image/train/{x}.jpg")
test_df["img_path"] = test_df["image_id"].apply(lambda x: f"/home/hwlee/dacon/imgQA/image/test/{x}.jpg")

train_df.to_csv("/home/hwlee/dacon/imgQA/preprocess_train.csv", index=False)
test_df.to_csv("/home/hwlee/dacon/imgQA/preprocess_test.csv", index=False)
