import os
from Training import training as train
from Training import training_balance_loss as train_balance
from Testing import testing as test

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    # modelName = ['Mobile'  'Alexnet'  'Inception'  'Tinydark'  'Mynet']
    # training(modelName='Alexnet', batch_size=64, learn_rate=0.0003, epochs=200, steps_per_epoch=1000,
    #          validation_steps=200, isprep=True)
    # training(modelName='Tinydark', batch_size=64, learn_rate=0.0003, epochs=200, steps_per_epoch=1000,
    #          validation_steps=200, isprep=True)
    # training(modelName='Inception', batch_size=64, learn_rate=0.0008, epochs=200, steps_per_epoch=1000,
    #          validation_steps=200, isprep=True)
    # train.training(modelName='Inception', batch_size=64, learn_rate=0.0008, epochs=200, steps_per_epoch=1000,
    #          validation_steps=200, isprep=False)
    # train_balance.training(modelName='Inception', batch_size=64, learn_rate=0.0008, epochs=200, steps_per_epoch=1000,
    #          validation_steps=200, isprep=True)

    saver_dir = r'D:\MyProject\GraduationProject\TensorFlow\models\Inception\2017-11-22_08-25-42'
    test.testing(modelName='Inception', fileset='test', batch_size=100, saver_dir=saver_dir)


