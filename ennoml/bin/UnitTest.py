"""
 Author : Utkarsh
"""

from ennoml.data import aug_fn, image_fn
from ennoml.models import EffNet
from ennoml.testing import testing_fn
from ennoml.training import Loading_Model, train_fn
import tensorflow as tf
import unittest
from PIL import Image
import numpy as np
import os, shutil

# The test based on unittest module
class TestEnnoML(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		# Creating a Image for testing
		self.image = Image.new("RGB", (200, 200), color='white')
		self.shape = self.image.size
		self.type = type(self.image)
		# This is required to be filled before testing.
		self.model_preTrained = r"D:\PROJECTS\Enlighten\Models\mobilenet\MaggiQR_MNetMk5A_2022_08_24_15"

	def test_augFN0(self):
		# Rotate Function
		self.image_rotate = aug_fn.image_aug().rotate_(self.image)
		self.rotate_shape = self.image_rotate.size
		self.assertEqual(self.shape, self.rotate_shape, 'Rotate Shape Does Not Match')
		self.assertEqual(self.type, type(self.image_rotate), 'Rotate Type Does Not Match')

	def test_augFN1(self):
		# Shift Function
		self.image_shift = aug_fn.image_aug().shift_(self.image)
		self.shift_shape = self.image_shift.size
		self.assertEqual(self.shape, self.shift_shape, 'Shift Shape Does Not Match')
		self.assertEqual(self.type, type(self.image_shift), 'Shift Type Does Not Match')

	def test_augFN2(self):
		# Resize Function
		self.image_resize = aug_fn.image_aug().resize_(self.image,100)
		self.resize_shape = self.image_resize.size
		self.assertEqual((100,100), self.resize_shape, 'resize Shape Does Not Match')
		self.assertEqual(self.type, type(self.image_resize), 'resize Type Does Not Match')

	def test_augFN3(self):
		# brightness Function
		self.image_brightness = aug_fn.image_aug().img_brightness(self.image)
		self.brightness_shape = self.image_brightness.size
		self.assertEqual(self.shape, self.brightness_shape, 'brightness Shape Does Not Match')
		self.assertEqual(self.type, type(self.image_brightness), 'brightness Type Does Not Match')

	def test_augFN4(self):
		# contrast Function
		self.image_contrast = aug_fn.image_aug().img_contrast(self.image)
		self.contrast_shape = self.image_contrast.size
		self.assertEqual(self.shape, self.contrast_shape, 'contrast Shape Does Not Match')
		self.assertEqual(self.type, type(self.image_contrast), 'contrast Type Does Not Match')


	def test_imageFN(self):
		# Loading Image into Tensor
		self.path = "test.jpg"
		self.image.save(self.path)
		self.image_load = image_fn.load_image(self.path,200)
		self.assertEqual((1, 200, 200, 3), self.image_load.shape, 'Image Loading Shape Does Not Match')
		self.assertEqual(np.ndarray, type(self.image_load), 'Image Loading Type Does Not Match')

	def test_models(self):
		# Loading EffNet Model Architecture
		self.model = EffNet.V1((300,300,3),3,include_top=False)
		self.assertEqual(tf.python.keras.engine.functional.Functional, type(self.model), 'Model EffNet Type Does Not Match')

	def test_trainings(self):
		# Loading MobileNet Architecture and  Compiling it
		self.model = Loading_Model.single(3,300,'',False).MobileNetV2(alpha=1)
		self.assertEqual(tf.python.keras.engine.sequential.Sequential, type(self.model), 'Model MobileNET Type Does Not Match')

	def test_util0(self):
		# Model Saving Function
		self.model_names = train_fn.model_save('test01','MobileNet')
		self.model_names_type = (type(self.model_names[0]), type(self.model_names[1]))
		self.assertEqual((str,str), self.model_names_type, 'Model Save Path Type Does Not Match')
		if os.path.exists(self.model_names[1]):
			shutil.rmtree(self.model_names[1])

	def test_model_loading(self):
		# Loading preTrained Model and Testing model to predict using a image.
		self.assertNotEqual(self.model_preTrained, "", 'Model Path is required for this test')
		self.assertTrue(os.path.exists(self.model_preTrained),'Given Model Path does not exists')
		if os.path.exists(self.model_preTrained):
			self.model_function = testing_fn.model_fn()
			self.model, self.label = self.model_function.load_model_file(self.model_preTrained)
			self.assertEqual(tf.python.keras.engine.sequential.Sequential, type(self.model), 'Model MobileNET Type Does Not Match')
			self.assertEqual(list, type(self.label), 'label Type Does Not Match')
			self.path = "test.jpg"
			self.image.save(self.path)
			self.image_load = np.asarray(Image.open(self.path))
			# Testing the model with image as a file
			self.product, self.prob = self.model_function.pred_flask(self.model, self.path)
			self.assertEqual(str, type(self.product), 'Prediction Output Type Does Not Match (File)')
			self.assertEqual(np.float32, type(self.prob), 'Prediction Probability Type Does Not Match (File)')
			# Testing the model with image as an array
			self.product1, self.prob1 = self.model_function.pred_flask(self.model, self.image_load , img_type='array')
			self.assertEqual(str, type(self.product1), 'Prediction Output Type Does Not Match (array)')
			self.assertEqual(np.float32, type(self.prob1), 'Prediction Probability Type Does Not Match (array)')


# run the test
unittest.main()