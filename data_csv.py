from util.data import Database
# 00 保存数据集为csv文件
save_ir_train_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_train.csv'
save_ir_test_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_test.csv'
ir_train_dir = '/home/wbo/Datasets/ir_train'
ir_test_dir = '/home/wbo/Datasets/ir_test'
Database(data_path=ir_train_dir, save_path=save_ir_train_csv)
Database(data_path=ir_test_dir, save_path=save_ir_test_csv)
