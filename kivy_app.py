from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Line
import numpy as np
import random
import os
import tensorflow as tf
from rdp import rdp
from sketch_rnn_train import *
from model import *
from utils import *
from rnn import *

#HELPER FUNCTIONS
def encode(input_strokes):
	strokes = to_big_strokes(input_strokes).tolist()
	strokes.insert(0, [0, 0, 1, 0, 0])
	seq_len = [len(input_strokes)]
	return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]

def decode(z_input=None, draw_mode=False, temperature=0.1, factor=0.2):
	z = None
	if z_input is not None:
		z = [z_input]
	sample_strokes, m = sample(sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=z)
	strokes = to_normal_strokes(sample_strokes)
	if draw_mode:
		draw_strokes(strokes, factor=factor)
	return strokes

#rdp simplification
#takes in [x,y,s] format
def rdp_lines(xy_lines,ep= 0.8):
	all_lines = []
	current_line = []
	for xy in xy_lines:
		x = int(xy[0])
		y = int(xy[1])
		eos = xy[2]
		current_line.append([x,y])
		if eos == 1:
			simple_line = np.array(rdp(current_line,epsilon=ep)) #higher epsilon = more simplified
			h,w = np.shape(simple_line)
			z = np.zeros((h,1),dtype = simple_line.dtype)
			simple_line = np.hstack((simple_line,z))
			h,w = np.shape(simple_line)
			simple_line[h-1,w-1] = 1
			if len(all_lines) == 0:
				all_lines = simple_line
			else:
				all_lines = np.concatenate((all_lines,simple_line))
			current_line = []
		else:
			continue
	h,w = np.shape(all_lines)
	z = np.zeros((h,1),dtype = all_lines.dtype)
	all_lines = np.hstack((all_lines,z))
	return all_lines

def lines_to_strokes(lines):
	lines[1:,0:2] -= lines[0:-1,0:2]
	lines[-1,3] = 1 # end of character
	lines[0] = [0, 0, 0, 0] # start at origin
	lines = lines[1:]
	return lines

#PAINT APPLICATION
class MyPaintWidget(Widget):
	num_strokes = 0
	stroke_data = []
	drawing_data = []

	def __init__(self, **kwargs):
		super(MyPaintWidget, self).__init__(**kwargs)

		#white background
		with self.canvas:
			self.rect = Rectangle(pos=self.pos, size=self.size)
		#bind size of window to widget
		self.bind(pos=self.update_rect)
		self.bind(size=self.update_rect)

		#initialize data variables w inputs if any

	def update_rect(self, *args):
		self.rect.pos = self.pos
		self.rect.size = self.size

	def on_touch_down(self, touch):
		with self.canvas:
			Color(0, 0, 0)
			touch.ud['line'] = Line(points=(touch.x, touch.y))
		self.num_strokes += 1
		self.stroke_data.append([int(touch.x), int(touch.y), 0])

	def on_touch_move(self, touch):
		self.stroke_data.append([int(touch.x), int(touch.y), 0])
		touch.ud['line'].points += [touch.x, touch.y]

	def on_touch_up(self, touch):
		#add final points
		self.stroke_data.append([int(touch.x), int(touch.y), 1])
		#reset and save data
		self.drawing_data.append(self.stroke_data)
		np.save("drawing_data.npy",np.array(self.drawing_data))
		#rdp simplification
		last_stroke = self.stroke_data
		h,w = np.shape(last_stroke)
		opt_ep = h / 50 #used to adjust simplification parameter, higher = more simple
		last_stroke_simple = rdp_lines(last_stroke, ep = 0.2)
		#point to 3-stroke format
		last_stroke_simple = lines_to_strokes(last_stroke_simple)
		print np.shape(last_stroke_simple)
		#encode stroke
		#temp_stroke = test_set.random_sample()
		z = encode(last_stroke_simple)
		z_strokes = decode(z)
		print(np.shape(z_strokes))
		self.stroke_data = []

	#draws a stroke from the machine
	def machine_draw(m_stroke):


class MyPaintApp(App):
    def build(self):
        return MyPaintWidget()


if __name__ == '__main__':
	model_dir = 'cat/'
	[hps_model, eval_hps_model, sample_hps_model] = load_model(model_dir)
	# construct the sketch-rnn model here:
	reset_graph()
	model = Model(hps_model)
	eval_model = Model(eval_hps_model, reuse=True)
	sample_model = Model(sample_hps_model, reuse=True)
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	# loads the weights from checkpoint into our model
	load_checkpoint(sess, model_dir)
	#load data
	data_dir = 'datasets/'
	[train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env(data_dir, model_dir)
	#cnn model only: [train_ims, valid_ims, test_ims] = load_env_ims(data_dir, model_dir)
	#run paint app
	MyPaintApp().run()







    