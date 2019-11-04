from keras.models import Model,load_model
from models.my_models import CenterVLAD
from keras.backend import l2_normalize, expand_dims

model_name = 'CenterVLAD'
model_path = '/home/wbo/PycharmProjects/Image_retrieval_qt/models/save_m/'+ model_name+ '.h5'
query_model_path = '/home/wbo/PycharmProjects/Image_retrieval_qt/pyqt5/query_models/query_' + model_name + '.h5'
if model_name == 'CenterVLAD' or 'ResNet50_CenterVLAD':
    base_model = load_model(model_path, custom_objects={'CenterVLAD': CenterVLAD,
                                                   'l2_normalize':l2_normalize,
                                                   'expand_dims':expand_dims})
else:
    base_model = load_model(model_path)

model = Model(inputs=base_model.input, outputs=[base_model.get_layer('feature').output, base_model.output])
print(model.summary())
model.save(query_model_path)