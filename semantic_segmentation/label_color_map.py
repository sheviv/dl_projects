# PyTorch FCN ResNet50 Model
#
# При сегментировании объектов на изображениях каждый из этих классов будет иметь различную цветовую маску.

# информация по картинке при передаче его в качестве входных данных для модели нейронной сети сегментации изображений
# outputs = model(image)
# print(type(outputs))
# пакет содержит вывод для одного изображения, 21 - модель дает результат для 21класса, на котором была обучена,
# последние два числа - высота и ширина
# print(outputs['out'].shape)
# многие тензоры имеют одинаковый выходной номер(все пиксели одного объекта классифицируются/маркируются одним номером)
# print(outputs)


# Создание списка цветовой карты
label_color_map = [
               (0, 0, 0),  # background
               (128, 0, 0),  # aeroplane
               (0, 128, 0),  # bicycle
               (128, 128, 0),  # bird
               (0, 0, 128),  # boat
               (128, 0, 128),  # bottle
               (0, 128, 128),  # bus
               (128, 128, 128),  # car
               (64, 0, 0),  # cat
               (192, 0, 0),  # chair
               (64, 128, 0),  # cow
               (192, 128, 0),  # dining table
               (64, 0, 128),  # dog
               (192, 0, 128),  # horse
               (64, 128, 128),  # motorbike
               (192, 128, 128),  # person
               (0, 64, 0),  # potted plant
               (128, 64, 0),  # sheep
               (0, 192, 0),  # sofa
               (128, 192, 0),  # train
               (0, 64, 128)  # tv/monitor
]