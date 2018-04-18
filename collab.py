from kivy.uix.popup import Popup
from kivy.uix.colorpicker import ColorPicker
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.app import App
from kivy.uix.button import Button
from kivy.properties import ListProperty 
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.graphics.opengl import glReadPixels
from kivy.core.image import Image as kImage
import numpy as np
import random
import os
import tensorflow as tf
from rdp import rdp
import svgwrite
from PIL import Image
col = [0,0,1,1]
#VAE MODEL
from sketch_rnn_train import *
from model_mod import *
from utils import *
from rnn import *
#CNN-LSTM MODEL
'''
from sketch_rnn_train_cnn_tf import *
from model_cnn_tf import *
from utils_cnn import *
from rnn import *
'''

#HELPER FUNCTIONS
def encode(input_strokes):
	strokes = to_big_strokes(input_strokes,max_len=189).tolist() #184 for collab2
	strokes.insert(0, [0, 0, 1, 0, 0])
	seq_len = [len(input_strokes)]
	return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]

def decode(z_input=None, draw_mode=False, temperature=0.1, factor=0.2):
	z = None
	if z_input is not None:
		z = [z_input]
	sample_strokes, m, _, _ = sample(sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=z)
	strokes = to_normal_strokes(sample_strokes)
	if draw_mode:
		draw_strokes(strokes, factor=factor)
	return strokes

def encode_cnn(input_ims):
	return sess.run(eval_model.batch_z, feed_dict={eval_model.input_ims: [input_ims]})[0]

def draw_strokes(data, color="black", doGradient=False, endColor="black", factor=0.2, svg_filename = 'test_drawing.svg'):
	tf.gfile.MakeDirs(os.path.dirname(svg_filename))
	min_x, max_x, min_y, max_y = get_bounds(data, factor)
	dims = (50 + max_x - min_x, 50 + max_y - min_y)
	dwg = svgwrite.Drawing(svg_filename, size=dims)
	dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
	lift_pen = 1
	abs_x = 25 - min_x 
	abs_y = 25 - min_y
	p = "M%s,%s " % (abs_x, abs_y)
	command = "m"
	for i in xrange(len(data)):
		if (lift_pen == 1):
			command = "m"
		elif (command != "l"):
			command = "l"
		else:
			command = ""
		x = float(data[i,0])/factor
		y = float(data[i,1])/factor
		lift_pen = data[i, 2]
		p += command+str(x)+","+str(y)+" "
	np.save("data_tester.npy",data)
	the_color = color
	stroke_width = 1
	dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
	dwg.save()

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
	try:
		h,w = np.shape(all_lines)
		z = np.zeros((h,1),dtype = all_lines.dtype)
		all_lines = np.hstack((all_lines,z))
		return all_lines
	except:
		return xy_lines

def lines_to_strokes(lines):
	lines = lines[:,:3]
	lines[1:,0:2] -= lines[0:-1,0:2]
	lines[-1,2] = 1 # end of drawing
	#lines[0] = [0, 0, 0] # start at origin
	#lines = lines[1:]
	return lines


class SelectedColorEllipse(Widget):
    selected_color = ListProperty(col)

class ColPckr(ColorPicker):
    pass

class ColPopup(Popup):
	pass

class Ex40(Widget):
    selected_color = ListProperty(col)
    def select_ColPckr(self,*args):
        ColPopup().open()
    def on_touch_down(self, touch):
        if touch.x <100 and touch.y < 100:
            return super(Ex40, self).on_touch_down(touch)
        sce = SelectedColorEllipse()
        sce.selected_color = self.selected_color
        sce.center = touch.pos
        self.add_widget(sce)

#PAINT APPLICATION
class MyPaintWidget(Widget):
	num_strokes = 0
	stroke_data = []
	drawing_data = []
	machine_mult_factor = 3
	save_frame_counter = 0
	pen_width = 1.0
	bg_array = np.array([])
	repeat_decode = 1
	collab = 0
	selected_color = Color(0,0,0)
	r = 0
	b = 0
	g = 0
	t = 255
	brush_count = 0
	texture0 = kImage('textures/brush0.jpg').texture

	def __init__(self, **kwargs):
		super(MyPaintWidget, self).__init__(**kwargs)
		texture0 = kImage('textures/brush0.jpg').texture
		#self.texture0.add_reload_observer(self.populate_texture)
		#white background
		with self.canvas:
			#Color(255 / 255.0,209 / 255.0,26 / 255.0)
			self.rect = Rectangle(pos=self.pos, size=self.size)
		#bind size of window to widget
		self.bind(pos=self.update_rect)
		self.bind(size=self.update_rect)

		self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
		self._keyboard.bind(on_key_down=self._on_keyboard_down)
		#initialize data variables w inputs if any

	def _keyboard_closed(self):
		self._keyboard.unbind(on_key_down=self._on_keyboard_down)
		self._keyboard = None

	def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
		print keycode
		if keycode[1] == 'n':
			self.canvas.clear()
			self.save_frame_counter = 0
			#white background
			with self.canvas:
				Color(255 / 255.0, 251 / 255.0, 253 / 255.0)
				self.rect = Rectangle(pos=self.pos, size=self.size)
		#change color
		elif keycode[0] == 49: #1
			self.r += 25
			print(self.r,self.g,self.b,self.t)
		elif keycode[0] == 50: #2
			self.g += 25
			print(self.r,self.g,self.b,self.t)
		elif keycode[0] == 51: #3
			self.b += 25
			print(self.r,self.g,self.b,self.t)
		elif keycode[0] == 52: #4
			self.t += 25
			print(self.r,self.g,self.b,self.t)
		elif keycode[0] == 53: #5
			self.r -= 25
			print(self.r,self.g,self.b,self.t)
		elif keycode[0] == 54: #6
			self.g -= 25
			print(self.r,self.g,self.b,self.t)
		elif keycode[0] == 55: #7
			self.b -= 25
			print(self.r,self.g,self.b,self.t)
		elif keycode[0] == 56: #8
			self.t -= 25
			print(self.r,self.g,self.b,self.t)
		elif keycode[0] == 57: #9
			self.r = 0
			self.g = 0
			self.b = 0
			self.t = 255
		#self.add_widget(Button())
		#adjust collab type
		elif keycode[1] == 'c':
			if self.collab == 2:
				self.collab = 0
			else:
				self.collab += 1
			print("collab_type=" + str(self.collab))
		#adjust decode times
		elif keycode[1] == 'd':
			self.repeat_decode += 1
		elif keycode[1] == 'f':
			self.repeat_decode -= 1 
		#adjust pen size
		elif keycode[0] == 61: #+
			self.pen_width += 0.25
			print(self.pen_width)
		elif keycode[0] == 45: #-
			self.pen_width -= 0.25
			print(self.pen_width)
			if self.pen_width <= 0:
				self.pen_width = 1.0
		elif keycode[1] == 'm':
			self.machine_mult_factor *= 2
		elif keycode[1] == 'b':
			self.machine_mult_factor /= 2
		elif keycode[1] == 't':
			if self.brush_count == 10:
				self.brush_count = 0
			else:
				self.brush_count += 1
			print("brush_count: " + str(self.brush_count))
		return True

	def update_rect(self, *args):
		self.rect.pos = self.pos
		self.rect.size = self.size

	def on_touch_down(self, touch):
		#texture testing
		brush_name = "textures/brush" + str(self.brush_count) + ".jpg"
		self.texture0 = kImage(brush_name).texture
		#Ellipse(texture = self.texture0, pos=(touch.x,touch.y), size=(self.pen_width,self.pen_width))
		#print(self.texture0)
		#print(self.size)
		if touch.y < 600 and touch.x > 0:
			with self.canvas:
				#simple line
				#Color(self.r / 255.0, self.g / 255.0, self.b / 255.0, self.t / 255.0)
				#touch.ud['line'] = Line(points=(touch.x, touch.y),width=self.pen_width, joint = 'bevel',cap = 'square')
				#with texture
				Color(1,1,1,0.5)
				touch.ud['line'] = Line(texture=self.texture0, points=(touch.x, touch.y),width=self.pen_width, joint = 'bevel',cap = 'square')

			self.num_strokes += 1
			self.stroke_data.append([int(touch.x), int(touch.y), 0])
		else:
			pass

	def on_touch_move(self, touch):
		if touch.y < 600 and touch.x > 000:
			self.stroke_data.append([int(touch.x), int(touch.y), 0])
			touch.ud['line'].points += [touch.x, touch.y]
			#with self.canvas:
			#	Color(1,1,1,0.5)
			#	Ellipse(texture = self.texture0, pos=(touch.x,touch.y), size=(self.pen_width,self.pen_width))
			#Window.screenshot(name="screenshots/"+str(self.save_frame_counter)+'.png')
			#self.save_frame_counter += 1
		else:
			pass

	#draws a stroke from the machine
	def machine_draw(self,m_stroke,x0,y0):
		#connect to last line
		'''
		xy_line = []
		xy_line.append(x0)
		xy_line.append(y0)
		last_x = x0
		last_y = y0
		for point in m_stroke:
			dx = point[0]
			dy = point[1]
			new_x = last_x + dx
			new_y = last_y + dy
			xy_line.append(new_x)
			xy_line.append(new_y)
			last_x = new_x
			last_y = new_y
		'''
		#disconnected line
		xy_line = []
		first_x = x0 + m_stroke[0][0] * 3
		first_y = y0 + m_stroke[0][1] * 3
		#first_x_mod = 800 - first_x
		xy_line.append(first_x)
		xy_line.append(first_y)
		last_x = first_x
		last_y = first_y
		for point in m_stroke[1:]:
			dx = point[0]
			dy = point[1]
			new_x = last_x + dx 
			new_y = last_y + dy
			xy_line.append(new_x)
			xy_line.append(new_y)
			last_x = new_x
			last_y = new_y
			#with self.canvas:
				#Color(1,1,1,0.5)
				#Ellipse(texture = self.texture0, pos=(new_x,new_y), size=(self.pen_width,self.pen_width))
		#draw stroke
		with self.canvas:
			#Color(self.r / 255.0, self.g / 255.0, self.b / 255.0, self.t / 255.0)
			Line(texture=self.texture0, points=xy_line,width = self.pen_width, joint = 'bevel',cap = 'square')
		#Window.screenshot(name="screenshots/"+str(self.save_frame_counter)+'.png')
		#self.save_frame_counter += 1

	def on_touch_up(self, touch, collab_type = 0):
		if touch.y < 600 and touch.x > 0:
			try:
				#Window.screenshot(name="undo.png")
				#Window.screenshot(name="screenshots/"+str(self.save_frame_counter)+'.png')
				#save_frame_counter += 1
				#add final points
				self.stroke_data.append([int(touch.x), int(touch.y), 1])
				#reset and save data
				self.drawing_data.append(self.stroke_data)
				#np.save("drawing_data.npy",np.array(self.drawing_data))
				#rdp simplification
				last_stroke = self.stroke_data
				if len(last_stroke) > 125:
					last_stroke = last_stroke[:125]
				h,w = np.shape(last_stroke)
				opt_ep = h / 50 #used to adjust simplification parameter, higher = more simple
				last_stroke_simple = rdp_lines(last_stroke, ep = 0.2)
				#print last_stroke_simple
				last_stroke_xs = last_stroke_simple[:,0]
				last_stroke_ys = last_stroke_simple[:,1]
				x_range = max(last_stroke_xs) - min(last_stroke_xs)
				y_range = max(last_stroke_ys) - min(last_stroke_ys)
				print x_range
				print y_range
				#point to 3-stroke format
				last_stroke_simple = lines_to_strokes(last_stroke_simple)
				last_strokes_5 = to_big_strokes(last_stroke_simple)
				#print last_stroke_simple
				#encode stroke
				#temp_stroke = test_set.random_sample()
				#collab_type = self.collab
				#collab_type = 2 for CNN only
				if collab_type == 0:
					for i in range(self.repeat_decode):
						#print("trying")
						#SIMPLE ENCODE DECODE
						z = encode(last_stroke_simple) #takes stroke 5 format
						z_strokes = decode(z)
						if x_range >= 200 or y_range >= 200:
							chosen_point_num = min(5,random.randint(1,len(z_strokes)))
							z_strokes = z_strokes[:chosen_point_num,:]
						else:
							z_strokes[:,:2] = self.machine_mult_factor * z_strokes[:,:2]
						#start at first point of original stroke
						first_x = last_stroke[0][0]
						first_y = last_stroke[0][1]
						#self.machine_draw(z_strokes,first_x,first_y)
						self.machine_draw(z_strokes,int(touch.x),int(touch.y))
				elif collab_type == 1:
					#encode human strokes to generate hidden state h
					for i in range(self.repeat_decode):
						_, hidden_states = get_hidden_states(sess=sess, model=sample_model, input_strokes = last_strokes_5)
						print(len(hidden_states))
						input_state = hidden_states[-1]
						z_strokes, _, _, _ = sample(sess=sess, model=sample_model, temperature=0.2, prev_state = input_state)
						z_strokes = to_normal_strokes(z_strokes)
						z_strokes[:,:2] = self.machine_mult_factor * z_strokes[:,:2]
						print("new_output")
						print(np.shape(z_strokes))
						draw_strokes(z_strokes)
						#SELECT STROKES
						#z_strokes = z_strokes[:,:10]
						stroke_col = z_strokes[:,2]
						stroke_ends = np.nonzero(stroke_col)[0]
						print(stroke_ends)
						if len(stroke_ends) > 3:
							#chosen_stroke_num = min(random.randint(0,5),len(stroke_ends) - 1)
							#print(chosen_stroke_num)
							#chosen_stroke_num = min(1,len(stroke_ends) - 1)
							start_stroke = stroke_ends[1]
							end_stroke = stroke_ends[3]
							#chosen_stroke_end = stroke_ends[] + 1
							#chosen_stroke_end = stroke_ends[chosen_stroke_num] + 1
							#chosen_z_strokes = z_strokes[:chosen_stroke_end,:]
							chosen_z_strokes = z_strokes[start_stroke:end_stroke,:]
						else:
							chosen_point_num = min(25,random.randint(1,len(z_strokes)))
							chosen_z_strokes = z_strokes[:chosen_point_num,:]
						#DRAW MACHINE STROKES
						first_x = last_stroke[0][0]
						first_y = last_stroke[0][1]
						#self.machine_draw(z_strokes,first_x,first_y)
						self.machine_draw(z_strokes,int(touch.x),int(touch.y))
						#self.machine_draw(chosen_z_strokes,first_x,first_y)
                        #self.machine_draw(chosen_z_strokes,int(touch.x),int(touch.y))
				#unconditional generation
				#FOR CNN MODELS
				elif collab_type == 3:
					#print("trying")
					z = np.random.randn(eval_model.hps.z_size)
					#print z
					z_strokes = decode(z)
					#print (np.shape(z_strokes))
					chosen_z_strokes[:,:2] = z_strokes[:,:2] / 2.0
					#self.machine_draw(chosen_z_strokes,int(touch.x),int(touch.y))
					self.machine_draw(z_strokes,int(touch.x),int(touch.y))
			except:
				pass
			self.stroke_data = []
			#self.export_to_png(filename=str(save_frame_counter)+'.png')
			#self.save_frame_counter += 1
			#Window.screenshot(name="screenshots/"+str(self.save_frame_counter)+'.png')
			#self.save_frame_counter += 1
		else:
			pass


class MyPaintApp(App):
    def build(self):
        return MyPaintWidget()

if __name__ == '__main__':
	#VAE MODEL
	model_dir = 'collab/'
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
	#data_dir = 'datasets/'
	#[train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env(data_dir, model_dir)
	#cnn model only: [train_ims, valid_ims, test_ims] = load_env_ims(data_dir, model_dir)
	#run paint app
	#Ex40App().run()
	MyPaintApp().run()
	#CNN-LSTM MODEL
	'''
	model_dir = 'face/'
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
	z = np.random.randn(eval_model.hps.z_size)
	z_strokes = decode(z)
	#z_strokes[:,:2] = 3 * z_strokes[:,:2]
	print(np.shape(z_strokes))
	#load data
	#data_dir = 'datasets/'
	#[train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env(data_dir, model_dir)
	#[train_ims, valid_ims, test_ims] = load_env_ims(data_dir, model_dir)
	#run paint app
	MyPaintApp().run()
	'''
	







    
